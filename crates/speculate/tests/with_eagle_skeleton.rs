//! EAGLE-2 skeleton integration test against the real published checkpoint.
//!
//! Downloads `yuhuili/EAGLE-LLaMA3-Instruct-8B/pytorch_model.bin` (1.5 GB)
//! and verifies that:
//!
//! 1. Our `EagleDraftConfig::eagle_llama3_8b` matches the published shape.
//! 2. `EagleDraftCandle::from_pth` parses every key in the checkpoint.
//! 3. A synthetic forward (random hidden state of the matching shape +
//!    a random token id) produces non-NaN output of the expected shape.
//!
//! This is a **format-compatibility** test, not a speedup benchmark — full
//! Llama 3 8B + EAGLE end-to-end needs a 24 GB+ GPU (8B BF16 ~16 GB +
//! EAGLE draft 1.5 GB > 16 GB), gated under `ABYO_LARGE_GPU=1` for that
//! follow-up measurement.
//!
//! Run with:
//! ```sh
//! cargo test --release --features cuda \
//!   -p abyo-speculate --test with_eagle_skeleton -- --ignored --nocapture
//! ```

#![cfg(not(target_os = "windows"))]

use abyo_speculate::methods::eagle::{EagleDraftCandle, EagleDraftConfig};
use abyo_speculate::model::hub::download_files;
use candle_core::{DType, Device, Tensor};

const REPO: &str = "yuhuili/EAGLE-LLaMA3-Instruct-8B";

fn pick_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    Device::Cpu
}

#[test]
#[ignore = "downloads ~1.5 GB EAGLE-LLaMA3 draft checkpoint"]
fn loads_real_eagle_llama3_8b_checkpoint() {
    let paths = download_files(REPO, &["pytorch_model.bin"]).expect("download EAGLE pth");
    assert_eq!(paths.len(), 1);

    let cfg = EagleDraftConfig::eagle_llama3_8b();
    let device = pick_device();
    let dtype = if matches!(device, Device::Cpu) {
        DType::F32
    } else {
        DType::F16 // EAGLE-LLaMA3 ships float16
    };

    let mut draft = EagleDraftCandle::from_pth(&cfg, &paths[0], &device, dtype)
        .expect("EagleDraftCandle::from_pth");
    assert_eq!(draft.config().hidden_size, 4096);
    assert_eq!(draft.config().vocab_size, 128256);

    // Synthetic forward: pretend we have a target hidden state for one token,
    // pretend the next token id is 42, position 0.
    let target_hidden = Tensor::randn(0f32, 0.02, (1, 1, 4096), &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let token_ids = Tensor::from_slice(&[42u32], (1, 1), &device).unwrap();

    let out = draft
        .forward(&target_hidden, &token_ids, 0)
        .expect("draft forward");
    assert_eq!(
        out.dims(),
        &[1, 1, 4096],
        "draft forward should produce [1, 1, hidden]"
    );

    // Verify non-NaN.
    let v: Vec<f32> = out
        .to_dtype(DType::F32)
        .unwrap()
        .reshape((4096,))
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(
        !v.iter().any(|x| x.is_nan() || x.is_infinite()),
        "draft forward produced NaN or Inf"
    );
    println!(
        "EAGLE-LLaMA3 draft forward: {} non-NaN finite values, mean = {:.4}",
        v.len(),
        v.iter().sum::<f32>() / v.len() as f32
    );
}
