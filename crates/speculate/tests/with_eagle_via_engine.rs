//! v0.5.0 — exercise the new SpeculateEngine dispatch for EAGLE-2.
//!
//! Same setup as `with_eagle_bf16_e2e.rs` but goes through the Engine
//! API:
//!
//! ```text
//! SpeculateEngine::builder()
//!     .target_model(...)
//!     .draft_model(...)
//!     .method(Method::Eagle2)
//!     .build()?
//!     .with_target(target_llama_decoder)
//!     .with_eagle_draft(eagle_candle)
//!     .with_eagle_run_config(EagleRunConfig { ... })
//!     .generate("[INST] ... [/INST]", 64)?
//! ```
//!
//! Run:
//! ```sh
//! cargo test --release --features cuda \
//!   -p abyo-speculate --test with_eagle_via_engine -- --ignored --nocapture
//! ```

#![cfg(not(target_os = "windows"))]

use abyo_speculate::methods::eagle::{EagleDraftCandle, EagleDraftConfig, EagleRunConfig};
use abyo_speculate::model::hub::{download_files, download_qwen2};
use abyo_speculate::model::llama::LlamaDecoder;
use abyo_speculate::model::llama_local::LlamaConfig;
use abyo_speculate::{Method, SpeculateEngine};
use candle_core::{DType, Device};

const TARGET_REPO: &str = "NousResearch/Llama-2-7b-chat-hf";
const EAGLE_REPO: &str = "yuhuili/EAGLE-llama2-chat-7B";

fn pick_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    Device::Cpu
}

#[test]
#[ignore = "downloads ~14 GB BF16 + ~1 GB EAGLE; needs 16 GB GPU"]
fn engine_dispatches_eagle2() {
    let (config_path, tokenizer_path, weights) =
        download_qwen2(TARGET_REPO).expect("download Llama-2-7B-Chat");
    let eagle_path = download_files(EAGLE_REPO, &["pytorch_model.bin"])
        .expect("download EAGLE")[0]
        .clone();

    let device = pick_device();
    let dtype = DType::BF16;

    let llama_cfg: LlamaConfig =
        serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
    let target = LlamaDecoder::from_paths(
        &llama_cfg.into_config(false),
        &weights,
        &tokenizer_path,
        device.clone(),
        dtype,
    )
    .expect("LlamaDecoder::from_paths");

    let draft = EagleDraftCandle::from_pth(
        &EagleDraftConfig::eagle_llama2_chat_7b(),
        &eagle_path,
        target.device(),
        dtype,
    )
    .expect("EagleDraftCandle::from_pth");

    let mut engine = SpeculateEngine::builder()
        .target_model(TARGET_REPO)
        .draft_model(EAGLE_REPO)
        .method(Method::Eagle2)
        .seed(12345)
        .build()
        .expect("build engine")
        .with_target(target)
        .with_eagle_draft(draft)
        .eagle_run_config(EagleRunConfig {
            top_k_per_step: 2,
            draft_depth: 2,
            ..EagleRunConfig::default()
        });

    assert!(engine.is_ready(), "engine should be ready post-attach");

    let out = engine
        .generate("[INST] Write a haiku about the ocean. [/INST]", 64)
        .expect("generate");
    eprintln!("--- engine EAGLE-2 output ---\n{out}");
    assert!(!out.is_empty());
}
