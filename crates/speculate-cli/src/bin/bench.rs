//! `abyo-speculate-bench` — measure tok/s for autoregressive vs vanilla SD
//! against real Qwen 2 / 2.5 checkpoints on the local GPU.
//!
//! Phase 2 scope: Qwen-only target/draft pairs. Generic backends land later.
//!
//! Example:
//!
//! ```sh
//! cargo run --release --features cuda --bin abyo-speculate-bench -- \
//!     --target Qwen/Qwen2.5-1.5B-Instruct \
//!     --draft  Qwen/Qwen2.5-0.5B-Instruct \
//!     --method both --max-tokens 128 --warmup 1 --runs 3
//! ```

use abyo_speculate::methods::vanilla::{run_vanilla_sd, VanillaConfig};
use abyo_speculate::model::hub::download_qwen2;
use abyo_speculate::model::qwen2::Qwen2Decoder;
use abyo_speculate::model::qwen2_local::Config;
use abyo_speculate::model::Decoder;
use abyo_speculate::sampling::{sample_from_distribution, softmax_with_temperature, top_p_filter};
use anyhow::{Context, Result};
use candle_core::{DType, Device};
use clap::{Parser, ValueEnum};
use rand::SeedableRng;

#[derive(Debug, Parser)]
#[command(name = "abyo-speculate-bench", version)]
struct Args {
    /// HF target model id (Qwen2 family with single safetensors shard).
    #[arg(long, default_value = "Qwen/Qwen2.5-1.5B-Instruct")]
    target: String,

    /// HF draft model id (typically a smaller Qwen2). Required for SD methods.
    #[arg(long, default_value = "Qwen/Qwen2.5-0.5B-Instruct")]
    draft: String,

    /// Which method(s) to bench.
    #[arg(long, value_enum, default_value_t = MethodArg::Both)]
    method: MethodArg,

    /// Tokens to generate per run.
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,

    /// Warm-up runs (excluded from timing).
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Measured runs.
    #[arg(long, default_value_t = 3)]
    runs: usize,

    /// Sampling temperature.
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// Top-p nucleus threshold (1.0 disables).
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,

    /// Number of draft tokens per SD round.
    #[arg(long, default_value_t = 4)]
    draft_lookahead: usize,

    /// Force CPU even if a CUDA / Metal device is available.
    #[arg(long)]
    cpu: bool,

    /// Optional prompt; defaults to a generic chat-style prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// RNG seed for deterministic sampling.
    #[arg(long, default_value_t = 12345)]
    seed: u64,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum MethodArg {
    /// Plain autoregressive baseline.
    Autoregressive,
    /// Vanilla SD (Leviathan 2023).
    Vanilla,
    /// Run both and report the speedup ratio.
    Both,
}

fn pick_device(force_cpu: bool) -> Device {
    if force_cpu {
        return Device::Cpu;
    }
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    #[cfg(feature = "metal")]
    if let Ok(d) = Device::new_metal(0) {
        return d;
    }
    Device::Cpu
}

fn load_qwen2(repo: &str, device: &Device, dtype: DType) -> Result<Qwen2Decoder> {
    let (config_path, tokenizer_path, weight_paths) =
        download_qwen2(repo).with_context(|| format!("downloading {repo}"))?;
    let config_json = std::fs::read_to_string(&config_path)?;
    let config: Config = serde_json::from_str(&config_json)
        .with_context(|| format!("parsing config.json from {repo}"))?;
    Qwen2Decoder::from_paths(
        &config,
        &weight_paths,
        &tokenizer_path,
        device.clone(),
        dtype,
    )
    .with_context(|| format!("loading Qwen2Decoder for {repo}"))
}

/// Plain autoregressive sampling loop with the same temperature / top-p
/// settings the SD path uses, so the comparison is apples-to-apples.
fn run_autoregressive(
    target: &mut Qwen2Decoder,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    temperature: f32,
    top_p: f32,
    seed: u64,
) -> anyhow::Result<Vec<u32>> {
    target.reset();
    Decoder::observe(target, prompt_ids)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(max_new_tokens);
    for _ in 0..max_new_tokens {
        let logits = Decoder::next_logits(target)?;
        let mut probs = softmax_with_temperature(&logits, temperature)?;
        if top_p < 1.0 {
            top_p_filter(&mut probs, top_p)?;
        }
        let tok = sample_from_distribution(&mut rng, &probs)? as u32;
        Decoder::observe(target, &[tok])?;
        out.push(tok);
    }
    Ok(out)
}

#[derive(Debug, Default, Clone)]
struct RunResult {
    tokens_generated: usize,
    elapsed_secs: f64,
}

impl RunResult {
    fn tok_per_sec(&self) -> f64 {
        if self.elapsed_secs > 0.0 {
            self.tokens_generated as f64 / self.elapsed_secs
        } else {
            0.0
        }
    }
}

fn run_method_n_times(
    label: &str,
    n: usize,
    mut once: impl FnMut() -> anyhow::Result<RunResult>,
) -> anyhow::Result<Vec<RunResult>> {
    let mut results = Vec::with_capacity(n);
    for i in 0..n {
        let r = once()?;
        eprintln!(
            "  [{label}] run {}: {} tokens in {:.3}s = {:.2} tok/s",
            i + 1,
            r.tokens_generated,
            r.elapsed_secs,
            r.tok_per_sec()
        );
        results.push(r);
    }
    Ok(results)
}

fn summary(label: &str, results: &[RunResult]) {
    if results.is_empty() {
        return;
    }
    let total_tokens: usize = results.iter().map(|r| r.tokens_generated).sum();
    let total_secs: f64 = results.iter().map(|r| r.elapsed_secs).sum();
    let mean = total_tokens as f64 / total_secs;
    let per_run: Vec<f64> = results.iter().map(|r| r.tok_per_sec()).collect();
    let min = per_run.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = per_run.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "[{label}] mean={mean:.2} tok/s | min={min:.2} | max={max:.2} | n={}",
        results.len()
    );
}

fn default_prompt() -> String {
    "Write a short, clear summary of how speculative decoding accelerates language model \
     inference, in plain English suitable for a senior software engineer who has not \
     specialized in ML. Cover the draft+verify pattern and where it falls short."
        .to_string()
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse();
    let device = pick_device(args.cpu);
    let dtype = if matches!(device, Device::Cpu) {
        DType::F32
    } else {
        DType::BF16
    };
    eprintln!(
        "device={:?} dtype={:?} target={} draft={}",
        device, dtype, args.target, args.draft
    );

    let prompt_text = args.prompt.unwrap_or_else(default_prompt);
    let max_new = args.max_tokens;
    let temperature = args.temperature;
    let top_p = args.top_p;
    let seed = args.seed;
    let lookahead = args.draft_lookahead;
    let warmup = args.warmup;
    let runs = args.runs;

    eprintln!("loading target...");
    let mut target = load_qwen2(&args.target, &device, dtype)?;
    let prompt_ids = target.encode(&prompt_text, true)?;
    eprintln!("prompt: {} tokens", prompt_ids.len());

    let do_ar = !matches!(args.method, MethodArg::Vanilla);
    let do_sd = !matches!(args.method, MethodArg::Autoregressive);

    let mut ar_summary: Option<f64> = None;
    let mut sd_summary: Option<f64> = None;

    if do_ar {
        eprintln!("\n=== autoregressive baseline ===");
        for w in 0..warmup {
            eprintln!("  [ar] warmup {}", w + 1);
            let _ =
                run_autoregressive(&mut target, &prompt_ids, max_new, temperature, top_p, seed)?;
        }
        let results = run_method_n_times("ar", runs, || {
            let t0 = std::time::Instant::now();
            let out =
                run_autoregressive(&mut target, &prompt_ids, max_new, temperature, top_p, seed)?;
            Ok(RunResult {
                tokens_generated: out.len(),
                elapsed_secs: t0.elapsed().as_secs_f64(),
            })
        })?;
        summary("ar", &results);
        ar_summary =
            Some(results.iter().map(|r| r.tok_per_sec()).sum::<f64>() / results.len() as f64);
    }

    if do_sd {
        eprintln!("\nloading draft...");
        let mut draft = load_qwen2(&args.draft, &device, dtype)?;
        let cfg = VanillaConfig {
            draft_lookahead: lookahead,
            temperature,
            top_p,
        };

        eprintln!("\n=== vanilla SD ===");
        for w in 0..warmup {
            eprintln!("  [sd] warmup {}", w + 1);
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let _ = run_vanilla_sd(
                &mut target,
                &mut draft,
                &prompt_ids,
                max_new,
                &cfg,
                &mut rng,
            )?;
        }
        let results = run_method_n_times("sd", runs, || {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let t0 = std::time::Instant::now();
            let out = run_vanilla_sd(
                &mut target,
                &mut draft,
                &prompt_ids,
                max_new,
                &cfg,
                &mut rng,
            )?;
            Ok(RunResult {
                tokens_generated: out.len(),
                elapsed_secs: t0.elapsed().as_secs_f64(),
            })
        })?;
        summary("sd", &results);
        sd_summary =
            Some(results.iter().map(|r| r.tok_per_sec()).sum::<f64>() / results.len() as f64);
    }

    if let (Some(ar), Some(sd)) = (ar_summary, sd_summary) {
        let speedup = sd / ar;
        eprintln!(
            "\n=== speedup ===\nautoregressive : {ar:.2} tok/s\nvanilla SD     : {sd:.2} tok/s\nratio          : {speedup:.2}x"
        );
        println!(
            r#"{{"target":"{}","draft":"{}","ar_tok_per_sec":{:.4},"sd_tok_per_sec":{:.4},"sd_speedup":{:.4},"max_tokens":{},"draft_lookahead":{},"temperature":{},"top_p":{}}}"#,
            args.target, args.draft, ar, sd, speedup, max_new, lookahead, temperature, top_p
        );
    }

    Ok(())
}
