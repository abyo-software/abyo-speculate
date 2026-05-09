//! Real-model integration tests against `microsoft/Phi-3-mini-4k-instruct`.
//!
//! Phi-3 mini is open access, ~3.8B params, ~7.6 GB BF16 — fits comfortably
//! on a 16 GB GPU and exercises the vendored `phi3_local` path (fused QKV
//! projection, fused gate+up MLP).
//!
//! Run with:
//! ```sh
//! cargo test --release --features cuda \
//!   -p abyo-speculate --test with_phi3_mini -- --ignored --nocapture
//! ```

#![cfg(not(target_os = "windows"))]

use abyo_speculate::model::hub::download_qwen2; // sharded auto-detector works for any HF repo
use abyo_speculate::model::phi3::Phi3Decoder;
use abyo_speculate::model::phi3_local::Config;
use abyo_speculate::model::Decoder;
use abyo_speculate::tree::DraftTree;
use candle_core::{DType, Device};

const REPO: &str = "microsoft/Phi-3-mini-4k-instruct";

fn pick_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    Device::Cpu
}

fn load_decoder() -> Phi3Decoder {
    let (config_path, tokenizer_path, weight_paths) =
        download_qwen2(REPO).expect("download Phi-3 mini");
    let config_json = std::fs::read_to_string(&config_path).unwrap();
    let config: Config = serde_json::from_str(&config_json).unwrap();
    let device = pick_device();
    let dtype = if matches!(device, Device::Cpu) {
        DType::F32
    } else {
        DType::BF16
    };
    Phi3Decoder::from_paths(&config, &weight_paths, &tokenizer_path, device, dtype)
        .expect("Phi3Decoder::from_paths")
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
#[ignore = "downloads ~7.6 GB and requires GPU for tolerable speed"]
fn autoregressive_smoke() {
    let mut dec = load_decoder();
    let prompt = dec.encode("The capital of France is", true).unwrap();
    Decoder::observe(&mut dec, &prompt).unwrap();

    let mut generated = Vec::new();
    for _ in 0..16 {
        let logits = Decoder::next_logits(&mut dec).unwrap();
        let tok = argmax_u32(&logits);
        generated.push(tok);
        Decoder::observe(&mut dec, &[tok]).unwrap();
    }
    let text = dec.decode(&generated, true).unwrap();
    println!("Phi-3 autoregressive: {text}");
    assert!(!text.trim().is_empty());
}

#[test]
#[ignore = "downloads ~7.6 GB and requires GPU for tolerable speed"]
fn tree_logits_match_per_path_observation_for_linear_tree() {
    let mut dec = load_decoder();
    let prompt = dec.encode("Once upon a time", true).unwrap();
    Decoder::observe(&mut dec, &prompt).unwrap();

    let mut drafts = Vec::new();
    for _ in 0..4 {
        let logits = Decoder::next_logits(&mut dec).unwrap();
        let tok = argmax_u32(&logits);
        drafts.push(tok);
        Decoder::observe(&mut dec, &[tok]).unwrap();
    }
    Decoder::rollback_to(&mut dec, prompt.len()).unwrap();

    let batched = Decoder::batched_logits(&mut dec, &drafts).unwrap();
    Decoder::rollback_to(&mut dec, prompt.len()).unwrap();

    let last_committed = *prompt.last().unwrap();
    let tree = DraftTree::linear(last_committed, &drafts);
    let tree_out = dec.tree_logits(&tree).unwrap();
    assert_eq!(tree_out.len(), tree.len());

    for i in 0..batched.len() {
        let am_b = argmax_u32(&batched[i]);
        let am_t = argmax_u32(&tree_out[i]);
        assert_eq!(
            am_b, am_t,
            "Phi-3 linear-tree row {i}: batched argmax {am_b} vs tree argmax {am_t}"
        );
    }
    assert_eq!(dec.history().len(), prompt.len());
}
