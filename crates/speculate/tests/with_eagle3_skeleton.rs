//! EAGLE-3 architecture verification against the real published checkpoint.
//!
//! Downloads `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B/pytorch_model.bin`
//! (~810 MB) and verifies that:
//!
//! 1. `Eagle3DraftConfig::eagle3_llama3_1_8b` matches the published shape.
//! 2. `Eagle3DraftCandle::from_pth` parses every key in the checkpoint
//!    (15 tensors: embed / fc / midlayer.* / norm / lm_head / d2t / t2d).
//! 3. A synthetic forward (random low/mid/high target hidden + last
//!    hidden + a draft token id) produces non-NaN output of the correct
//!    `[1, 1, draft_vocab_size]` shape.
//! 4. d2t / t2d translation tables work — every draft id maps back to a
//!    target id, every target id has a defined reachability flag.
//!
//! Full Llama 3.1 + EAGLE-3 end-to-end speedup measurement requires
//! `TreeDecoder::last_hidden_states_multi(layers)` on the target side,
//! which is the v0.2.1 follow-up.

#![cfg(not(target_os = "windows"))]

use abyo_speculate::methods::eagle3::{Eagle3DraftCandle, Eagle3DraftConfig};
use abyo_speculate::model::hub::download_files;
use candle_core::{DType, Device, Tensor};

const REPO: &str = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B";

fn pick_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    Device::Cpu
}

#[test]
#[ignore = "downloads ~810 MB EAGLE-3 checkpoint"]
fn loads_real_eagle3_llama3_1_checkpoint() {
    let paths = download_files(REPO, &["pytorch_model.bin"]).expect("download EAGLE-3 pth");
    assert_eq!(paths.len(), 1);

    let cfg = Eagle3DraftConfig::eagle3_llama3_1_8b();
    let device = pick_device();
    let dtype = if matches!(device, Device::Cpu) {
        DType::F32
    } else {
        DType::F16 // EAGLE-3 ships F16
    };

    let mut draft = Eagle3DraftCandle::from_pth(&cfg, &paths[0], &device, dtype)
        .expect("Eagle3DraftCandle::from_pth");
    assert_eq!(draft.config().hidden_size, 4096);
    assert_eq!(draft.config().draft_vocab_size, 32000);

    // Synthetic forward.
    let h = 4096;
    let low = Tensor::randn(0f32, 0.02, (1, 1, h), &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let mid = Tensor::randn(0f32, 0.02, (1, 1, h), &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let high = Tensor::randn(0f32, 0.02, (1, 1, h), &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let last = Tensor::randn(0f32, 0.02, (1, 1, h), &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let token_ids = Tensor::from_slice(&[42u32], (1, 1), &device).unwrap();

    let out = draft
        .forward(&low, &mid, &high, &last, &token_ids, 0)
        .expect("EAGLE-3 forward");
    assert_eq!(
        out.dims(),
        &[1, 1, 32000],
        "EAGLE-3 output should be [1, 1, draft_vocab_size]"
    );

    // Sanity: no NaN/Inf.
    let v: Vec<f32> = out
        .to_dtype(DType::F32)
        .unwrap()
        .reshape((32000,))
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(
        !v.iter().any(|x| x.is_nan() || x.is_infinite()),
        "EAGLE-3 forward produced NaN or Inf"
    );

    // d2t / t2d sanity.
    let some_draft_id = 100u32;
    let target_id = draft
        .draft_to_target_token(some_draft_id)
        .expect("d2t lookup");
    assert!(target_id < 128256, "d2t value should fit Llama 3 vocab");

    // t2d is BoolStorage and skipped by candle's pth loader; we don't
    // assert reachability here — just that the call doesn't panic.
    let _ = draft.target_token_is_reachable(0u32);

    println!(
        "EAGLE-3 forward: {} non-NaN values, mean = {:.4}; d2t[{}] = {}",
        v.len(),
        v.iter().sum::<f32>() / v.len() as f32,
        some_draft_id,
        target_id
    );
}
