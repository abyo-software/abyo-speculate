//! EAGLE-2 against a BF16 Llama target. The "use EAGLE in production"
//! starter — load the official `yuhuili/EAGLE-llama2-chat-7B` checkpoint
//! against BF16 Llama-2-7B-Chat and run greedy generation with EOS
//! detection.
//!
//! ```sh
//! cargo run --release --features cuda --example eagle2_bf16
//! ```
//!
//! ~15 GB total GPU footprint (fits a 16 GB card). First run downloads
//! ~14 GB of safetensors + ~1 GB of EAGLE weights and caches them under
//! `~/.cache/huggingface/hub`.
//!
//! Important caveats — also covered in [`README.md`](../../../README.md)
//! and [`BENCHMARKS.md`](../../../BENCHMARKS.md):
//!
//! - On 16 GB consumer GPUs we measure EAGLE-2 at ~0.5× of plain
//!   autoregressive throughput on Llama 2 7B BF16. The architecture is
//!   MHA, per-step AR is already cheap, and per-round SD overhead
//!   doesn't amortize. EAGLE shines on bigger / GQA / >24 GB setups.
//! - Greedy acceptance with deeper-tree (depth ≥ 2) verification can
//!   produce trajectories that differ slightly from the AR baseline
//!   past the first 1-2 rounds (multi-position GEMM precision drift on
//!   `per_node_logits[i > 0]`; the root row is patched via the v0.2.2
//!   GEMV-path replacement). Output is still semantically valid.
//!
//! Use this example as a template, then tune `EagleRunConfig::draft_depth`
//! / `top_k_per_step` / `max_tree_nodes` for your target architecture.

use abyo_speculate::methods::eagle::{
    run_eagle, EagleDraftCandle, EagleDraftConfig, EagleRunConfig,
};
use abyo_speculate::model::hub::{download_files, download_qwen2};
use abyo_speculate::model::llama::LlamaDecoder;
use abyo_speculate::model::llama_local::LlamaConfig;
use abyo_speculate::model::Decoder;
use anyhow::Result;
use candle_core::{DType, Device};
use rand::SeedableRng;

fn pick_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    Device::Cpu
}

fn main() -> Result<()> {
    let target_repo = "NousResearch/Llama-2-7b-chat-hf";
    let eagle_repo = "yuhuili/EAGLE-llama2-chat-7B";

    println!("downloading {target_repo} (first run only, ~14 GB)...");
    let (config_path, tokenizer_path, weights) = download_qwen2(target_repo)?;
    println!("downloading {eagle_repo} (first run only, ~1 GB)...");
    let eagle_path = download_files(eagle_repo, &["pytorch_model.bin"])?[0].clone();

    let device = pick_device();
    let dtype = DType::BF16;

    let llama_cfg: LlamaConfig = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
    let mut target = LlamaDecoder::from_paths(
        &llama_cfg.into_config(false),
        &weights,
        &tokenizer_path,
        device.clone(),
        dtype,
    )?;
    println!(
        "target loaded: {} layers, hidden={}, vocab={}",
        target.num_hidden_layers(),
        target.hidden_size(),
        Decoder::vocab_size(&target),
    );

    let mut draft = EagleDraftCandle::from_pth(
        &EagleDraftConfig::eagle_llama2_chat_7b(),
        &eagle_path,
        target.device(),
        dtype,
    )?;

    let prompt_text = "[INST] Write a haiku about the ocean. [/INST]";
    let prompt = target.encode(prompt_text, true)?;
    println!("\nprompt ({} tokens): {prompt_text}", prompt.len());

    let cfg = EagleRunConfig {
        top_k_per_step: 2,
        draft_depth: 2,
        max_tree_nodes: None,
        temperature: 0.0,
        top_p: 1.0,
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let t0 = std::time::Instant::now();
    let out = run_eagle(&mut target, &mut draft, &prompt, 128, &cfg, &mut rng)?;
    let secs = t0.elapsed().as_secs_f64();

    let text = target.decode(&out, true)?;
    println!(
        "\n--- EAGLE-2 output ({} tokens in {:.2}s = {:.1} tok/s) ---\n{text}",
        out.len(),
        secs,
        out.len() as f64 / secs,
    );
    Ok(())
}
