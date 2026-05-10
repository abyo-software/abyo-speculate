//! Regression test for the v0.2.1 finding that
//! `LlamaQuantDecoder::tree_logits(multi-node tree)` returns a different
//! root distribution than `next_logits` would at the same state.
//!
//! For correctness, the following invariant **must** hold:
//!   `argmax(tree_logits(tree)[0]) == argmax(next_logits())`
//! whenever `tree.token_at(0) == history.last()`.
//!
//! Empirical observation (v0.2.1):
//! - 1-node tree: invariant holds.
//! - 31-node Cartesian tree on Llama 3.1 8B Q4_K_M: invariant **fails**
//!   (root argmax shifts from 264 = " a" to 12366 = " Paris").
//! - EAGLE-2 + Llama 3.0 8B Q4_K_M: output matches AR (the same
//!   underlying multi-position bug exists but the argmax happens to be
//!   stable for that prompt — true acceptance rate is hard to assess).
//!
//! This test is gated under `#[ignore]` so the suite stays green; running
//! it on a CUDA box is the v0.2.2 reproduction harness.

#![cfg(not(target_os = "windows"))]

use abyo_speculate::methods::medusa::{MedusaConfig, MedusaHeads, TreeTopology};
use abyo_speculate::model::hub::download_files;
use abyo_speculate::model::quantized_llama::LlamaQuantDecoder;
use abyo_speculate::model::{Decoder, TreeDecoder};
use abyo_speculate::tree::DraftTree;
use candle_core::Device;

const GGUF_REPO: &str = "QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF";
const GGUF_FILE: &str = "Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf";
const TOKENIZER_REPO: &str = "NousResearch/Meta-Llama-3.1-8B-Instruct";
const LLAMA31_EOS: &[u32] = &[128001, 128009];

fn pick_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    Device::Cpu
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i, x)
            } else {
                (bi, bv)
            }
        })
        .0
}

#[test]
#[ignore = "downloads ~4.9 GB Llama 3.1 GGUF; v0.2.2 will fix the underlying bug"]
fn tree_logits_root_must_match_next_logits() {
    let gguf = download_files(GGUF_REPO, &[GGUF_FILE]).expect("download GGUF")[0].clone();
    let tok = download_files(TOKENIZER_REPO, &["tokenizer.json"])
        .expect("download tokenizer")[0]
        .clone();
    let mut t = LlamaQuantDecoder::from_gguf(&gguf, &tok, pick_device(), LLAMA31_EOS.to_vec())
        .expect("LlamaQuantDecoder::from_gguf");

    let prompt = t.encode("The capital of France is", true).unwrap();
    Decoder::observe(&mut t, &prompt).unwrap();
    let nl_argmax = argmax(&Decoder::next_logits(&mut t).unwrap());
    let root = *t.history().last().unwrap();

    // 1-node tree (just the root).
    let t1 = DraftTree::from_parent_table(&[(0, root)]).unwrap();
    let l1 = TreeDecoder::tree_logits(&mut t, &t1).unwrap();
    let l1_argmax = argmax(&l1[0]);
    assert_eq!(
        l1_argmax, nl_argmax,
        "1-node tree_logits[0] must match next_logits"
    );

    // 31-node Cartesian (k=2 depth=4).
    let cart = MedusaHeads::from_config(MedusaConfig {
        n_heads: 4,
        hidden_size: 4096,
        vocab_size: 128_256,
        residual_layers: 1,
    })
    .build_draft_tree(
        root,
        &[
            vec![100, 200],
            vec![300, 400],
            vec![500, 600],
            vec![700, 800],
        ],
        TreeTopology::CartesianProduct,
    )
    .unwrap();
    let l_cart = TreeDecoder::tree_logits(&mut t, &cart).unwrap();
    let l_cart_argmax = argmax(&l_cart[0]);
    // **THIS WILL FAIL** in v0.2.1 — kept as a regression marker for v0.2.2.
    assert_eq!(
        l_cart_argmax, nl_argmax,
        "31-node tree_logits[0] must match next_logits (currently fails — see eagle3.rs notes)"
    );
}
