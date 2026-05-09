//! Streaming vanilla Speculative Decoding example.
//!
//! ```sh
//! cargo run --release --features cuda --example vanilla_sd_streaming
//! ```
//!
//! Pairs Qwen 2.5 3B Instruct (target) with Qwen 2.5 0.5B Instruct (draft),
//! streams generated tokens to stdout as they're emitted, stops on EOS.
//! On RTX 4070 Ti SUPER this runs ~1.4× faster than the autoregressive
//! baseline of the same target.

use abyo_speculate::model::hub::download_qwen2;
use abyo_speculate::model::qwen2::Qwen2Decoder;
use abyo_speculate::model::qwen2_local::Config;
use abyo_speculate::model::Decoder;
use abyo_speculate::{GenerationOptions, Method, SpeculateEngine};
use anyhow::Result;
use candle_core::{DType, Device};
use std::io::Write;

fn pick_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    Device::Cpu
}

fn load(repo: &str, device: &Device, dtype: DType) -> Result<Qwen2Decoder> {
    let (config_path, tokenizer_path, weight_paths) = download_qwen2(repo)?;
    let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
    Ok(Qwen2Decoder::from_paths(
        &config,
        &weight_paths,
        &tokenizer_path,
        device.clone(),
        dtype,
    )?)
}

fn main() -> Result<()> {
    let target_repo = "Qwen/Qwen2.5-3B-Instruct";
    let draft_repo = "Qwen/Qwen2.5-0.5B-Instruct";
    let device = pick_device();
    let dtype = if matches!(device, Device::Cpu) {
        DType::F32
    } else {
        DType::BF16
    };

    println!("loading target {target_repo}…");
    let target = load(target_repo, &device, dtype)?;
    println!("loading draft  {draft_repo}…");
    let draft = load(draft_repo, &device, dtype)?;
    let prompt_text =
        "Explain in one paragraph why Pure Rust speculative decoding matters for local LLMs.";
    let prompt_ids = target.encode(prompt_text, true)?;
    let stops = target.eos_token_ids();

    let mut engine = SpeculateEngine::builder()
        .target_model(target_repo)
        .draft_model(draft_repo)
        .method(Method::Vanilla)
        .draft_lookahead(4)
        .seed(123)
        .build()?
        .with_target(target)
        .with_draft(draft);

    print!("\n{prompt_text}");
    std::io::stdout().flush()?;

    let opts = GenerationOptions::new(128).with_stops(stops);

    // Streaming callback: print each token as we go. We need a tokenizer
    // here for incremental detokenization — for now we just print the raw
    // ids; replace with target.decode(&[tok], true) for prettier output.
    let _out = engine.generate_tokens_with(&prompt_ids, &opts, |tok| {
        print!(" [{tok}]");
        std::io::stdout().flush().ok();
        true
    })?;
    println!();
    Ok(())
}
