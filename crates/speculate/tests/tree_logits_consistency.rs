//! Regression test for the v0.2.1 finding that
//! `LlamaQuantDecoder::tree_logits(multi-node tree)` returns a different
//! root distribution than `next_logits` would at the same state.
//!
//! For correctness, the following invariant **must** hold:
//!   `argmax(tree_logits(tree)[0]) == argmax(next_logits())`
//! whenever `tree.token_at(0) == history.last()`.
//!
//! v0.2.1 isolated this as a Q4 GEMV-vs-GEMM precision drift in
//! `forward_with_positions`: single-position calls go through candle's
//! GEMV kernel (matches `next_logits` bit-for-bit); multi-position calls
//! go through the GEMM path with a different FP accumulation order, and
//! the resulting per-token logit values drift by ~0.01–0.05 — enough to
//! flip a borderline argmax (e.g. ` a` vs ` Paris` differ by 0.02 at the
//! end of "The capital of France is").
//!
//! v0.2.2 fix in `LlamaQuantDecoder::tree_logits`: overwrite
//! `per_node_logits[0]` with the GEMV-path logits captured at the
//! restoration `forward_advance_logits([last_committed])`. The root row
//! is then guaranteed to match `next_logits`; deeper rows still go
//! through the GEMM path but are only consulted after the corresponding
//! draft token is accepted (i.e. matches root's argmax).

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

    // Bisect tree size: build linear chains of length k and check root.
    // Sweep tree sizes 2..=32 (linear chains so positions vary in depth).
    for n in [2, 3, 4, 5, 8, 16, 32] {
        let mut entries: Vec<(usize, u32)> = vec![(0, root)];
        for i in 1..n {
            entries.push((i - 1, 1000 + i as u32));
        }
        let tk = DraftTree::from_parent_table(&entries).unwrap();
        let lk = TreeDecoder::tree_logits(&mut t, &tk).unwrap();
        assert_eq!(
            argmax(&lk[0]),
            nl_argmax,
            "{n}-node linear tree_logits[0] must match next_logits"
        );
    }

    // 31-node Cartesian (the EAGLE-2 default tree).
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
    assert_eq!(
        argmax(&l_cart[0]),
        nl_argmax,
        "31-node Cartesian tree_logits[0] must match next_logits"
    );
    let _ = l1_argmax;
}
