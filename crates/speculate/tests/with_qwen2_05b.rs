//! Real-model integration tests against `Qwen/Qwen2.5-0.5B-Instruct`.
//!
//! Gated `#[ignore]` because they:
//! - Need network access (or a pre-warmed `hf-hub` cache).
//! - Pull ~1 GB of weights on first run.
//! - Require a working CUDA / Metal stack (CPU works but is very slow).
//!
//! Run with:
//! ```sh
//! cargo test --release --features cuda \
//!   -p abyo-speculate --test with_qwen2_05b -- --ignored --nocapture
//! ```
//!
//! These are end-to-end correctness checks for the `Qwen2Decoder` /
//! `qwen2_local` glue. They do **not** assert specific generated text
//! (deterministic-but-irrelevant under greedy sampling); they assert that
//! shapes / cache state remain consistent and that the three forward paths
//! (autoregressive, batched, tree) agree on overlapping logits.

#![cfg(not(target_os = "windows"))] // hf-hub on Windows uses a different cache layout we don't test

use abyo_speculate::model::hub::download_qwen2_single_shard;
use abyo_speculate::model::qwen2::Qwen2Decoder;
use abyo_speculate::model::qwen2_local::Config;
use abyo_speculate::model::Decoder;
use abyo_speculate::tree::DraftTree;
use candle_core::{DType, Device};

const REPO: &str = "Qwen/Qwen2.5-0.5B-Instruct";

fn pick_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    Device::Cpu
}

fn load_decoder() -> Qwen2Decoder {
    let (config_path, tokenizer_path, weights_path) =
        download_qwen2_single_shard(REPO).expect("download Qwen2.5 0.5B");
    let config_json = std::fs::read_to_string(&config_path).unwrap();
    let config: Config = serde_json::from_str(&config_json).unwrap();
    let device = pick_device();
    // Use F32 on CPU (safer dynamic range), BF16 on GPU (close to native fmt).
    let dtype = if matches!(device, Device::Cpu) {
        DType::F32
    } else {
        DType::BF16
    };
    Qwen2Decoder::from_paths(&config, &[weights_path], &tokenizer_path, device, dtype)
        .expect("Qwen2Decoder::from_paths")
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
#[ignore = "downloads ~1GB and requires GPU for tolerable speed"]
fn autoregressive_smoke() {
    let mut dec = load_decoder();
    let prompt = dec.encode("The capital of France is", true).unwrap();
    assert!(!prompt.is_empty());

    Decoder::observe(&mut dec, &prompt).unwrap();
    let mut generated = Vec::new();
    for _ in 0..16 {
        let logits = Decoder::next_logits(&mut dec).unwrap();
        let tok = argmax_u32(&logits);
        generated.push(tok);
        Decoder::observe(&mut dec, &[tok]).unwrap();
    }
    let text = dec.decode(&generated, true).unwrap();
    println!("autoregressive output: {text}");
    assert!(!text.trim().is_empty());
}

#[test]
#[ignore = "downloads ~1GB and requires GPU for tolerable speed"]
fn batched_logits_match_sequential_observation() {
    // The k+1 logits returned by batched_logits must match the logits we
    // would get by running next_logits after observing each prefix in turn.
    let mut dec = load_decoder();
    let prompt = dec
        .encode("Hello, world! Today the weather is", true)
        .unwrap();
    Decoder::observe(&mut dec, &prompt).unwrap();

    // Pick deterministic drafts: target's own greedy continuation.
    let mut drafts = Vec::new();
    for _ in 0..3 {
        let logits = Decoder::next_logits(&mut dec).unwrap();
        let tok = argmax_u32(&logits);
        drafts.push(tok);
        Decoder::observe(&mut dec, &[tok]).unwrap();
    }
    // Roll back so we can re-call batched_logits from the same starting point.
    Decoder::rollback_to(&mut dec, prompt.len()).unwrap();

    // Sequential (observe + next_logits, rolling back between).
    let mut sequential: Vec<Vec<f32>> = Vec::new();
    for i in 0..=drafts.len() {
        let baseline_len = prompt.len();
        Decoder::rollback_to(&mut dec, baseline_len).unwrap();
        if i > 0 {
            Decoder::observe(&mut dec, &drafts[..i]).unwrap();
        }
        sequential.push(Decoder::next_logits(&mut dec).unwrap());
    }

    // Batched.
    Decoder::rollback_to(&mut dec, prompt.len()).unwrap();
    let batched = Decoder::batched_logits(&mut dec, &drafts).unwrap();
    assert_eq!(batched.len(), drafts.len() + 1);

    // Compare argmax (cheap and robust to BF16 noise).
    for (i, (a, b)) in batched.iter().zip(sequential.iter()).enumerate() {
        let am = argmax_u32(a);
        let bm = argmax_u32(b);
        assert_eq!(am, bm, "batched vs sequential argmax disagree at row {i}");
    }
}

#[test]
#[ignore = "downloads ~1GB and requires GPU for tolerable speed"]
fn tree_logits_match_per_path_observation_for_linear_tree() {
    // For a linear tree (no branching), tree_logits must agree with
    // batched_logits on the linear sequence. This exercises forward_with_positions
    // + truncate_kv_cache_to + the 4D attention bias plumbing.
    let mut dec = load_decoder();
    let prompt = dec.encode("Once upon a time", true).unwrap();
    Decoder::observe(&mut dec, &prompt).unwrap();

    // Pick drafts greedily, rollback between.
    let mut drafts = Vec::new();
    for _ in 0..4 {
        let logits = Decoder::next_logits(&mut dec).unwrap();
        let tok = argmax_u32(&logits);
        drafts.push(tok);
        Decoder::observe(&mut dec, &[tok]).unwrap();
    }
    Decoder::rollback_to(&mut dec, prompt.len()).unwrap();

    // Batched-linear baseline.
    let batched = Decoder::batched_logits(&mut dec, &drafts).unwrap();
    Decoder::rollback_to(&mut dec, prompt.len()).unwrap();

    // Tree (linear) — root = last committed prompt token, tail = drafts.
    let last_committed = *prompt.last().unwrap();
    let tree = DraftTree::linear(last_committed, &drafts);
    let tree_out = dec.tree_logits(&tree).unwrap();
    assert_eq!(tree_out.len(), tree.len(), "one logit row per tree node");

    // tree_out[0] corresponds to "after the root", which equals batched[0].
    // tree_out[i for i>=1] corresponds to "after path-to-i", which equals batched[i].
    for i in 0..batched.len() {
        let am_batched = argmax_u32(&batched[i]);
        let am_tree = argmax_u32(&tree_out[i]);
        assert_eq!(
            am_batched, am_tree,
            "linear-tree row {i}: batched argmax {am_batched} vs tree argmax {am_tree}"
        );
    }

    // Cache state must be unchanged after tree_logits.
    assert_eq!(dec.history().len(), prompt.len());
}
