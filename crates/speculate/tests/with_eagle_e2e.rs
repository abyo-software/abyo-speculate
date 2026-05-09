//! Real EAGLE-2 end-to-end speedup measurement.
//!
//! Pairs:
//! - Llama 3 8B Instruct **Q4_K_M GGUF** (target, ~4.9 GB) via
//!   `LlamaQuantDecoder`.
//! - `yuhuili/EAGLE-LLaMA3-Instruct-8B` (1.5 GB EAGLE-2 draft).
//!
//! Total GPU footprint: ~7 GB, fits comfortably on a 16 GB card. (BF16
//! Llama 3 8B + EAGLE wouldn't — see `tests/with_real_medusa_e2e.rs` for
//! the gated 24 GB+ path.)
//!
//! Run with:
//! ```sh
//! cargo test --release --features cuda \
//!   -p abyo-speculate --test with_eagle_e2e -- --ignored --nocapture
//! ```

#![cfg(not(target_os = "windows"))]

use abyo_speculate::methods::eagle::{
    run_eagle, EagleDraftCandle, EagleDraftConfig, EagleRunConfig,
};
use abyo_speculate::model::hub::download_files;
use abyo_speculate::model::quantized_llama::LlamaQuantDecoder;
use abyo_speculate::model::Decoder;
use candle_core::{DType, Device};
use rand::SeedableRng;

const GGUF_REPO: &str = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF";
const GGUF_FILE: &str = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf";
const TOKENIZER_REPO: &str = "NousResearch/Meta-Llama-3-8B-Instruct";
const EAGLE_REPO: &str = "yuhuili/EAGLE-LLaMA3-Instruct-8B";

// Llama 3 EOS: 128001 (end_of_text), 128009 (eot_id).
const LLAMA3_EOS: &[u32] = &[128001, 128009];

fn pick_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    Device::Cpu
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
#[ignore = "downloads ~4.9 GB Llama 3 Q4 GGUF + 1.5 GB EAGLE checkpoint"]
fn llama3_8b_q4_with_eagle_2_speedup() {
    // 1. Download all weights.
    let gguf_path = download_files(GGUF_REPO, &[GGUF_FILE]).expect("download GGUF")[0].clone();
    let tokenizer_path =
        download_files(TOKENIZER_REPO, &["tokenizer.json"]).expect("download tokenizer")[0].clone();
    let eagle_path =
        download_files(EAGLE_REPO, &["pytorch_model.bin"]).expect("download EAGLE")[0].clone();

    let device = pick_device();
    let dtype = DType::F16; // EAGLE-LLaMA3 ships F16

    // 2. Load target.
    eprintln!("loading Llama 3 8B Q4_K_M target...");
    let mut target = LlamaQuantDecoder::from_gguf(
        &gguf_path,
        &tokenizer_path,
        device.clone(),
        LLAMA3_EOS.to_vec(),
    )
    .expect("LlamaQuantDecoder::from_gguf");

    // 3. Load EAGLE draft.
    eprintln!("loading EAGLE-2 draft...");
    let cfg = EagleDraftConfig::eagle_llama3_8b();
    let mut draft = EagleDraftCandle::from_pth(&cfg, &eagle_path, &device, dtype)
        .expect("EagleDraftCandle::from_pth");

    // 4. Encode prompt.
    let prompt_text = "The capital of France is";
    let prompt_ids = target.encode(prompt_text, true).unwrap();
    eprintln!("prompt: {} tokens", prompt_ids.len());

    // 5. Autoregressive baseline.
    eprintln!("\n=== autoregressive baseline ===");
    Decoder::observe(&mut target, &prompt_ids).unwrap();
    let n = 32usize;
    let mut ar_out = Vec::with_capacity(n);
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        let logits = Decoder::next_logits(&mut target).unwrap();
        let tok = argmax_u32(&logits);
        ar_out.push(tok);
        Decoder::observe(&mut target, &[tok]).unwrap();
    }
    let ar_secs = t0.elapsed().as_secs_f64();
    let ar_text = target.decode(&ar_out, true).unwrap();
    eprintln!(
        "AR: {n} tokens in {ar_secs:.3}s = {:.2} tok/s",
        n as f64 / ar_secs
    );
    eprintln!("AR text: {ar_text}");

    // 6. EAGLE-2 with Cartesian then dynamic tree.
    eprintln!("\n=== EAGLE-2 (Cartesian, 4×2 = 31 nodes) ===");
    let cart_cfg = EagleRunConfig {
        top_k_per_step: 2,
        draft_depth: 4,
        max_tree_nodes: None,
        temperature: 0.0,
        top_p: 1.0,
    };
    let (cart_secs, cart_n) = {
        let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
        let t1 = std::time::Instant::now();
        let out = run_eagle(
            &mut target,
            &mut draft,
            &prompt_ids,
            n,
            &cart_cfg,
            &mut rng,
        )
        .unwrap_or_else(|e| {
            eprintln!("Cart run failed: {e:?}");
            Vec::new()
        });
        let secs = t1.elapsed().as_secs_f64();
        let text = if out.is_empty() {
            "(empty)".to_string()
        } else {
            target.decode(&out, true).unwrap()
        };
        eprintln!(
            "Cart: {} tokens in {secs:.3}s = {:.2} tok/s",
            out.len(),
            out.len() as f64 / secs
        );
        eprintln!("Cart text: {text}");
        (secs, out.len())
    };

    eprintln!("\n=== EAGLE-2 dynamic (depth=4 k=2 pruned to 16 nodes) ===");
    let dyn_cfg = EagleRunConfig {
        top_k_per_step: 2,
        draft_depth: 4,
        max_tree_nodes: Some(16),
        temperature: 0.0,
        top_p: 1.0,
    };
    let (dyn_secs, dyn_n) = {
        let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
        let t1 = std::time::Instant::now();
        let out = run_eagle(
            &mut target,
            &mut draft,
            &prompt_ids,
            n,
            &dyn_cfg,
            &mut rng,
        )
        .unwrap_or_else(|e| {
            eprintln!("Dyn run failed: {e:?}");
            Vec::new()
        });
        let secs = t1.elapsed().as_secs_f64();
        let text = if out.is_empty() {
            "(empty)".to_string()
        } else {
            target.decode(&out, true).unwrap()
        };
        eprintln!(
            "Dyn16: {} tokens in {secs:.3}s = {:.2} tok/s",
            out.len(),
            out.len() as f64 / secs
        );
        eprintln!("Dyn16 text: {text}");
        (secs, out.len())
    };

    let cart_speedup = if cart_n > 0 {
        (cart_n as f64 / cart_secs) / (n as f64 / ar_secs)
    } else {
        0.0
    };
    let dyn_speedup = if dyn_n > 0 {
        (dyn_n as f64 / dyn_secs) / (n as f64 / ar_secs)
    } else {
        0.0
    };
    println!(
        r#"{{"target":"Meta-Llama-3-8B-Instruct.Q4_K_M","draft":"EAGLE-LLaMA3-Instruct-8B","method":"eagle-2","ar_tok_per_sec":{:.4},"sd_cart_tok_per_sec":{:.4},"sd_cart_speedup":{:.4},"sd_dyn16_tok_per_sec":{:.4},"sd_dyn16_speedup":{:.4},"max_tokens":{n},"draft_depth":{},"top_k_per_step":{}}}"#,
        n as f64 / ar_secs,
        cart_n as f64 / cart_secs.max(1e-9),
        cart_speedup,
        dyn_n as f64 / dyn_secs.max(1e-9),
        dyn_speedup,
        cart_cfg.draft_depth,
        cart_cfg.top_k_per_step,
    );
    eprintln!(
        "\n=== EAGLE-2 speedup: Cart {cart_speedup:.2}× | Dyn16 {dyn_speedup:.2}× over AR ==="
    );

    assert!(ar_out.len() == n);
    assert!(cart_n > 0 || dyn_n > 0);
}
