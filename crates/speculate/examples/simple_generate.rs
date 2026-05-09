//! Minimal text-in / text-out example.
//!
//! ```sh
//! cargo run --release --features cuda --example simple_generate
//! ```
//!
//! Loads Qwen 2.5 0.5B Instruct, generates a short completion, prints it.
//! No SD; just exercises the engine's autoregressive path against a real
//! checkpoint. ~1 GB download on first run, cached afterwards.

use abyo_speculate::model::hub::download_qwen2;
use abyo_speculate::model::qwen2::Qwen2Decoder;
use abyo_speculate::model::qwen2_local::Config;
use abyo_speculate::{Method, SpeculateEngine};
use anyhow::Result;
use candle_core::{DType, Device};

fn pick_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    Device::Cpu
}

fn main() -> Result<()> {
    let repo = "Qwen/Qwen2.5-0.5B-Instruct";
    let (config_path, tokenizer_path, weight_paths) = download_qwen2(repo)?;
    let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;

    let device = pick_device();
    let dtype = if matches!(device, Device::Cpu) {
        DType::F32
    } else {
        DType::BF16
    };
    let target = Qwen2Decoder::from_paths(&config, &weight_paths, &tokenizer_path, device, dtype)?;

    let mut engine = SpeculateEngine::builder()
        .target_model(repo)
        .method(Method::Autoregressive)
        .seed(42)
        .build()?
        .with_target(target);

    let out = engine.generate("The capital of France is", 32)?;
    println!("{out}");
    Ok(())
}
