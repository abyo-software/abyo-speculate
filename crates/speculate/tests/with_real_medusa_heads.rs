//! Real-Medusa-head loader test against `FasterDecoding/medusa-vicuna-7b-v1.3`.
//!
//! Downloads only the `medusa_lm_head.pt` file (~1.4 GB) — no 14 GB Vicuna
//! base model — and validates that:
//!
//! 1. `MedusaHeadsCandle::from_fasterdecoding_pt` parses the published key
//!    layout (`<i>.<j>.linear.{weight,bias}` + `<i>.<num_layers>.weight`).
//! 2. The loaded heads' forward pass produces correctly-shaped, non-NaN
//!    logits when given a synthetic hidden state of the matching shape.
//! 3. `top_k_per_head` returns indices within the model's vocab.
//!
//! This is a **format-compatibility** test, not a speedup benchmark — random
//! hidden states yield meaningless predictions, but the loader and forward
//! plumbing are exercised against a real published checkpoint.
//!
//! Run with:
//! ```sh
//! cargo test --release --features cuda \
//!   -p abyo-speculate --test with_real_medusa_heads -- --ignored --nocapture
//! ```

#![cfg(not(target_os = "windows"))]

use abyo_speculate::methods::medusa::{MedusaConfig, MedusaHeadsCandle};
use abyo_speculate::model::hub::download_files;
use candle_core::{DType, Device, Tensor};

const REPO: &str = "FasterDecoding/medusa-vicuna-7b-v1.3";

fn pick_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    Device::Cpu
}

#[test]
#[ignore = "downloads ~1.4GB Medusa head .pt file"]
fn loads_fasterdecoding_medusa_head_v1_3() {
    let paths = download_files(REPO, &["medusa_lm_head.pt"]).expect("download head");
    assert_eq!(paths.len(), 1);

    // Vicuna 7B v1.3 / Llama-2 architecture: hidden 4096, vocab 32000.
    // The file actually stores 5 heads (config.json's `medusa_num_heads: 2`
    // is outdated for this checkpoint — trust the file).
    let cfg = MedusaConfig {
        n_heads: 5,
        hidden_size: 4096,
        vocab_size: 32000,
        residual_layers: 1,
    };
    let device = pick_device();
    let dtype = DType::BF16;
    let heads = MedusaHeadsCandle::from_fasterdecoding_pt(&cfg, &paths[0], &device, dtype)
        .expect("from_fasterdecoding_pt");
    assert_eq!(heads.config().n_heads, 5);

    // Synthetic hidden state, matching shape Vicuna's last hidden state would
    // have. We don't assert prediction quality — only that shapes and dtypes
    // line up and the math doesn't NaN out.
    let hidden = Tensor::randn(0f32, 1f32, 4096, &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    let logits_per_head = heads.forward(&hidden).unwrap();
    assert_eq!(logits_per_head.len(), 5, "one logit tensor per head");
    for (i, l) in logits_per_head.iter().enumerate() {
        let dims = l.dims();
        assert_eq!(dims, &[32000], "head {i} logits should be [vocab]");
        // Materialize and sanity-check.
        let v: Vec<f32> = l.to_dtype(DType::F32).unwrap().to_vec1().unwrap();
        let any_nan = v.iter().any(|x| x.is_nan());
        let any_inf = v.iter().any(|x| x.is_infinite());
        assert!(!any_nan, "head {i} produced NaN logits");
        assert!(!any_inf, "head {i} produced Inf logits");
    }

    // top_k_per_head: indices stay within vocab.
    let top3 = heads.top_k_per_head(&hidden, 3).unwrap();
    assert_eq!(top3.len(), 5);
    for (i, p) in top3.iter().enumerate() {
        assert_eq!(p.len(), 3, "head {i} should yield 3 candidates");
        for &t in p {
            assert!(t < 32000, "head {i} produced out-of-vocab token {t}");
        }
    }
}
