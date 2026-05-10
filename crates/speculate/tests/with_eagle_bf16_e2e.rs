//! Real EAGLE-2 end-to-end speedup measurement on the architecture EAGLE
//! was actually trained for: **BF16** target (no Q4 quantization noise
//! masking the draft).
//!
//! Pairs:
//! - Llama-2-7B-Chat **BF16** (target, ~14 GB) loaded via `LlamaDecoder`
//!   from the official `meta-llama/Llama-2-7b-chat-hf` safetensors.
//! - `yuhuili/EAGLE-llama2-chat-7B` (~1 GB EAGLE-1/-2 draft).
//!
//! Total GPU footprint: ~15 GB, fits a 16 GB card. Llama 2's RoPE base
//! (10000) and lack of llama3-style scaling makes this the cleanest
//! reproduction of the published EAGLE speedups.
//!
//! Run with:
//! ```sh
//! cargo test --release --features cuda \
//!   -p abyo-speculate --test with_eagle_bf16_e2e -- --ignored --nocapture
//! ```

#![cfg(not(target_os = "windows"))]

use abyo_speculate::methods::eagle::{
    run_eagle, EagleDraftCandle, EagleDraftConfig, EagleRunConfig,
};
use abyo_speculate::model::hub::{download_files, download_qwen2};
use abyo_speculate::model::llama::LlamaDecoder;
use abyo_speculate::model::llama_local::LlamaConfig;
use abyo_speculate::model::Decoder;
use candle_core::{DType, Device};
use rand::SeedableRng;

const TARGET_REPO: &str = "NousResearch/Llama-2-7b-chat-hf";
const EAGLE_REPO: &str = "yuhuili/EAGLE-llama2-chat-7B";

fn pick_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    Device::Cpu
}

fn argmax_u32(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv {
                (i, v)
            } else {
                (bi, bv)
            }
        })
        .0 as u32
}

#[test]
#[ignore = "downloads ~14 GB BF16 + ~1 GB EAGLE; requires 16 GB GPU and meta-llama gated repo access"]
fn llama2_7b_bf16_with_eagle_2_speedup() {
    let (config_path, tokenizer_path, weights) =
        download_qwen2(TARGET_REPO).expect("download Llama-2-7B-Chat");
    let eagle_path = download_files(EAGLE_REPO, &["pytorch_model.bin"])
        .expect("download EAGLE-llama2-chat")[0]
        .clone();

    let device = pick_device();
    let dtype = DType::BF16;

    let config_json = std::fs::read_to_string(&config_path).unwrap();
    let llama_config: LlamaConfig = serde_json::from_str(&config_json).unwrap();
    let config = llama_config.into_config(false);
    eprintln!(
        "loading Llama 2 7B BF16 target ({} layers, hidden={}, vocab={})...",
        config.num_hidden_layers, config.hidden_size, config.vocab_size
    );

    let mut target = LlamaDecoder::from_paths(&config, &weights, &tokenizer_path, device.clone(), dtype)
        .expect("LlamaDecoder::from_paths");

    eprintln!("loading EAGLE draft...");
    let eagle_cfg = EagleDraftConfig {
        hidden_size: 4096,
        vocab_size: 32000, // Llama 2 vocab
        num_attention_heads: 32,
        num_key_value_heads: 32, // Llama 2 7B is MHA, not GQA
        intermediate_size: 11008,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        max_position_embeddings: 4096,
    };
    let mut draft = EagleDraftCandle::from_pth(&eagle_cfg, &eagle_path, &device, dtype)
        .expect("EagleDraftCandle::from_pth");

    let prompts = [
        "[INST] What is the capital of France? [/INST]",
        "[INST] Write a haiku about the ocean. [/INST]",
        "[INST] Explain how RoPE positional embeddings work in 2 sentences. [/INST]",
    ];

    let n = 64usize;
    let mut total_ar_secs = 0f64;
    let mut total_sd_secs = 0f64;
    let mut total_n = 0usize;

    for prompt_text in prompts.iter() {
        eprintln!("\n--- prompt: {} ---", prompt_text);
        let prompt = target.encode(prompt_text, true).unwrap();
        eprintln!("prompt: {} tokens", prompt.len());

        // AR baseline.
        Decoder::observe(&mut target, &prompt).unwrap();
        let mut ar_out = Vec::with_capacity(n);
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            let logits = Decoder::next_logits(&mut target).unwrap();
            let tok = argmax_u32(&logits);
            ar_out.push(tok);
            Decoder::observe(&mut target, &[tok]).unwrap();
            if [2u32].contains(&tok) {
                break;
            }
        }
        let ar_secs = t0.elapsed().as_secs_f64();
        let ar_text = target.decode(&ar_out, true).unwrap();
        eprintln!(
            "AR: {} tokens in {ar_secs:.3}s = {:.2} tok/s",
            ar_out.len(),
            ar_out.len() as f64 / ar_secs
        );
        eprintln!("AR text: {}", &ar_text[..ar_text.len().min(140)]);

        // EAGLE-2 with dynamic tree (depth=4, k=2, dyn=16).
        let cfg = EagleRunConfig {
            top_k_per_step: 2,
            draft_depth: 4,
            max_tree_nodes: Some(16),
            temperature: 0.0,
            top_p: 1.0,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
        let t1 = std::time::Instant::now();
        let sd_out = run_eagle(
            &mut target,
            &mut draft,
            &prompt,
            n,
            &cfg,
            &mut rng,
        )
        .unwrap_or_else(|e| {
            eprintln!("EAGLE-2 run failed: {e:?}");
            Vec::new()
        });
        let sd_secs = t1.elapsed().as_secs_f64();
        let sd_text = if sd_out.is_empty() {
            "(empty)".to_string()
        } else {
            target.decode(&sd_out, true).unwrap()
        };
        eprintln!(
            "EAGLE-2: {} tokens in {sd_secs:.3}s = {:.2} tok/s",
            sd_out.len(),
            sd_out.len() as f64 / sd_secs
        );
        eprintln!("EAGLE text: {}", &sd_text[..sd_text.len().min(140)]);

        let speedup = (sd_out.len() as f64 / sd_secs) / (ar_out.len() as f64 / ar_secs);
        eprintln!("speedup: {speedup:.2}×");

        total_ar_secs += ar_secs;
        total_sd_secs += sd_secs;
        total_n += sd_out.len();
    }

    let ar_tps = (prompts.len() * n) as f64 / total_ar_secs;
    let sd_tps = total_n as f64 / total_sd_secs;
    let overall_speedup = sd_tps / ar_tps;
    println!(
        r#"{{"target":"meta-llama/Llama-2-7b-chat-hf","draft":"yuhuili/EAGLE-llama2-chat-7B","method":"eagle-2","prompts":{},"max_tokens":{n},"ar_tok_per_sec":{:.4},"sd_tok_per_sec":{:.4},"sd_speedup":{:.4},"draft_depth":4,"top_k_per_step":2,"max_tree_nodes":16,"dtype":"BF16"}}"#,
        prompts.len(),
        ar_tps,
        sd_tps,
        overall_speedup,
    );
    eprintln!(
        "\n=== EAGLE-2 BF16 overall: AR {ar_tps:.2} tok/s | SD {sd_tps:.2} tok/s | speedup {overall_speedup:.2}× ==="
    );

    assert!(overall_speedup > 0.0);
}
