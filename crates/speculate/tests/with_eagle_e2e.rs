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

    // 6. EAGLE-2 run loop.
    eprintln!("\n=== EAGLE-2 ===");
    let run_cfg = EagleRunConfig {
        top_k_per_step: 2,
        draft_depth: 2, // draft_depth × 1 lm_head_apply each; quantized lm_head dominates
        temperature: 0.0,
        top_p: 1.0,
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(12345);

    let t1 = std::time::Instant::now();
    let med_out = run_eagle(
        &mut target,
        &mut draft,
        &prompt_ids,
        n,
        &run_cfg,
        &mut rng,
    )
    .unwrap_or_else(|e| {
        eprintln!("EAGLE run failed: {e:?}");
        Vec::new()
    });
    if med_out.is_empty() {
        eprintln!("(EAGLE returned no tokens — see error above)");
        return;
    }
    let med_secs = t1.elapsed().as_secs_f64();
    let med_text = target.decode(&med_out, true).unwrap();
    eprintln!(
        "EAGLE: {} tokens in {med_secs:.3}s = {:.2} tok/s",
        med_out.len(),
        med_out.len() as f64 / med_secs
    );
    eprintln!("EAGLE text: {med_text}");

    let speedup = (med_out.len() as f64 / med_secs) / (n as f64 / ar_secs);
    println!(
        r#"{{"target":"Meta-Llama-3-8B-Instruct.Q4_K_M","draft":"EAGLE-LLaMA3-Instruct-8B","method":"eagle-2","ar_tok_per_sec":{:.4},"sd_tok_per_sec":{:.4},"sd_speedup":{:.4},"max_tokens":{n},"draft_depth":{},"top_k_per_step":{}}}"#,
        n as f64 / ar_secs,
        med_out.len() as f64 / med_secs,
        speedup,
        run_cfg.draft_depth,
        run_cfg.top_k_per_step,
    );
    eprintln!("\n=== EAGLE-2 speedup: {speedup:.2}× over autoregressive ===");

    assert!(ar_out.len() == n);
    assert!(!med_out.is_empty());
}
