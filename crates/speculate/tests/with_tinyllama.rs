//! Real-model integration tests against TinyLlama 1.1B Chat.
//!
//! Same correctness checks as `with_qwen2_05b.rs` but exercising the
//! vendored Llama path (`model::llama_local` + `model::llama::LlamaDecoder`).
//! TinyLlama is open-access (no HF token), Llama-2-architecture, ~2.2 GB
//! BF16 — small enough to ship as a default test target.
//!
//! Run with:
//! ```sh
//! cargo test --release --features cuda \
//!   -p abyo-speculate --test with_tinyllama -- --ignored --nocapture
//! ```

#![cfg(not(target_os = "windows"))]

use abyo_speculate::model::hub::download_qwen2; // sharded auto-detector works for any HF repo
use abyo_speculate::model::llama::LlamaDecoder;
use abyo_speculate::model::llama_local::LlamaConfig;
use abyo_speculate::model::Decoder;
use abyo_speculate::tree::DraftTree;
use candle_core::{DType, Device};

const REPO: &str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";

fn pick_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    Device::Cpu
}

fn load_decoder() -> LlamaDecoder {
    let (config_path, tokenizer_path, weight_paths) =
        download_qwen2(REPO).expect("download TinyLlama");
    let config_json = std::fs::read_to_string(&config_path).unwrap();
    let hf_config: LlamaConfig = serde_json::from_str(&config_json).unwrap();
    let config = hf_config.into_config(false); // flash_attn unsupported in our vendored version
    let device = pick_device();
    let dtype = if matches!(device, Device::Cpu) {
        DType::F32
    } else {
        DType::BF16
    };
    LlamaDecoder::from_paths(&config, &weight_paths, &tokenizer_path, device, dtype)
        .expect("LlamaDecoder::from_paths")
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
#[ignore = "downloads ~2.2GB and requires GPU for tolerable speed"]
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
    println!("Llama autoregressive: {text}");
    assert!(!text.trim().is_empty());
}

#[test]
#[ignore = "downloads ~2.2GB and requires GPU for tolerable speed"]
fn batched_logits_match_sequential_observation() {
    let mut dec = load_decoder();
    let prompt = dec
        .encode("Hello, world! Today the weather is", true)
        .unwrap();
    Decoder::observe(&mut dec, &prompt).unwrap();

    let mut drafts = Vec::new();
    for _ in 0..3 {
        let logits = Decoder::next_logits(&mut dec).unwrap();
        let tok = argmax_u32(&logits);
        drafts.push(tok);
        Decoder::observe(&mut dec, &[tok]).unwrap();
    }
    Decoder::rollback_to(&mut dec, prompt.len()).unwrap();

    let mut sequential: Vec<Vec<f32>> = Vec::new();
    for i in 0..=drafts.len() {
        Decoder::rollback_to(&mut dec, prompt.len()).unwrap();
        if i > 0 {
            Decoder::observe(&mut dec, &drafts[..i]).unwrap();
        }
        sequential.push(Decoder::next_logits(&mut dec).unwrap());
    }

    Decoder::rollback_to(&mut dec, prompt.len()).unwrap();
    let batched = Decoder::batched_logits(&mut dec, &drafts).unwrap();
    assert_eq!(batched.len(), drafts.len() + 1);

    for (i, (a, b)) in batched.iter().zip(sequential.iter()).enumerate() {
        let am = argmax_u32(a);
        let bm = argmax_u32(b);
        assert_eq!(am, bm, "batched vs sequential argmax disagree at row {i}");
    }
}

#[test]
#[ignore = "downloads ~2.2GB and requires GPU for tolerable speed"]
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
        let am_batched = argmax_u32(&batched[i]);
        let am_tree = argmax_u32(&tree_out[i]);
        assert_eq!(
            am_batched, am_tree,
            "Llama linear-tree row {i}: batched argmax {am_batched} vs tree argmax {am_tree}"
        );
    }
    assert_eq!(dec.history().len(), prompt.len());
}
