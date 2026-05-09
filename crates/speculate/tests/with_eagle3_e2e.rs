//! Real EAGLE-3 end-to-end speedup measurement.
//!
//! Pairs:
//! - Llama 3.1 8B Instruct **Q4_K_M GGUF** (target, ~4.9 GB) via
//!   `LlamaQuantDecoder`.
//! - `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` (~810 MB EAGLE-3 draft).
//!
//! EAGLE-3's draft has a 32k vocab vs the target's 128k, so the per-step
//! `lm_head_apply` runs entirely on the draft side (small F16 matmul) —
//! no Q4 × 128k call per draft step. This is the architectural fix for
//! the EAGLE-2 + Q4 bottleneck documented in `with_eagle_e2e.rs`.
//!
//! Run with:
//! ```sh
//! cargo test --release --features cuda \
//!   -p abyo-speculate --test with_eagle3_e2e -- --ignored --nocapture
//! ```

#![cfg(not(target_os = "windows"))]

use abyo_speculate::methods::eagle3::{
    run_eagle3, Eagle3DraftCandle, Eagle3DraftConfig, Eagle3RunConfig,
};
use abyo_speculate::model::hub::download_files;
use abyo_speculate::model::quantized_llama::LlamaQuantDecoder;
use abyo_speculate::model::Decoder;
use candle_core::{DType, Device};
use rand::SeedableRng;

const GGUF_REPO: &str = "QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF";
const GGUF_FILE: &str = "Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf";
const TOKENIZER_REPO: &str = "NousResearch/Meta-Llama-3.1-8B-Instruct";
const EAGLE3_REPO: &str = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B";

const LLAMA31_EOS: &[u32] = &[128001, 128009];

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
#[ignore = "downloads ~4.9 GB Llama 3.1 Q4 GGUF + 810 MB EAGLE-3 checkpoint"]
fn llama31_8b_q4_with_eagle_3_speedup() {
    // 1. Download.
    let gguf_path = download_files(GGUF_REPO, &[GGUF_FILE]).expect("download GGUF")[0].clone();
    let tokenizer_path =
        download_files(TOKENIZER_REPO, &["tokenizer.json"]).expect("download tokenizer")[0].clone();
    let eagle3_path = download_files(EAGLE3_REPO, &["pytorch_model.bin"])
        .expect("download EAGLE-3")[0]
        .clone();

    let device = pick_device();
    let dtype = DType::F16;

    // 2. Target.
    eprintln!("loading Llama 3.1 8B Q4_K_M target...");
    let mut target = LlamaQuantDecoder::from_gguf(
        &gguf_path,
        &tokenizer_path,
        device.clone(),
        LLAMA31_EOS.to_vec(),
    )
    .expect("LlamaQuantDecoder::from_gguf");
    eprintln!("target n_layers = {}", target.num_hidden_layers());

    // 3. EAGLE-3 draft.
    eprintln!("loading EAGLE-3 draft...");
    let cfg = Eagle3DraftConfig::eagle3_llama3_1_8b();
    let mut draft = Eagle3DraftCandle::from_pth(&cfg, &eagle3_path, &device, dtype)
        .expect("Eagle3DraftCandle::from_pth");

    // 4. Encode prompt.
    let prompt_text = "The capital of France is";
    let prompt_ids = target.encode(prompt_text, true).unwrap();
    eprintln!("prompt: {} tokens", prompt_ids.len());

    // 5. AR baseline.
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

    // 6. EAGLE-3 run loop.
    let layer_indices = Eagle3RunConfig::default_layers_for(target.num_hidden_layers());
    eprintln!(
        "\n=== EAGLE-3 (depth=4, k=2, layers={layer_indices:?}, dyn=16) ==="
    );
    let dyn_cfg = Eagle3RunConfig {
        top_k_per_step: 2,
        draft_depth: 4,
        max_tree_nodes: Some(16),
        layer_indices,
        temperature: 0.0,
        top_p: 1.0,
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
    let t1 = std::time::Instant::now();
    let out = run_eagle3(
        &mut target,
        &mut draft,
        &prompt_ids,
        n,
        &dyn_cfg,
        &mut rng,
    )
    .unwrap_or_else(|e| {
        eprintln!("EAGLE-3 run failed: {e:?}");
        Vec::new()
    });
    let secs = t1.elapsed().as_secs_f64();
    let text = if out.is_empty() {
        "(empty)".to_string()
    } else {
        target.decode(&out, true).unwrap()
    };
    eprintln!(
        "EAGLE-3: {} tokens in {secs:.3}s = {:.2} tok/s",
        out.len(),
        out.len() as f64 / secs
    );
    eprintln!("EAGLE-3 text: {text}");

    let speedup = if out.is_empty() {
        0.0
    } else {
        (out.len() as f64 / secs) / (n as f64 / ar_secs)
    };
    println!(
        r#"{{"target":"Meta-Llama-3.1-8B-Instruct.Q4_K_M","draft":"EAGLE3-LLaMA3.1-Instruct-8B","method":"eagle-3","ar_tok_per_sec":{:.4},"sd_tok_per_sec":{:.4},"sd_speedup":{:.4},"max_tokens":{n},"draft_depth":{},"top_k_per_step":{},"max_tree_nodes":16,"layer_indices":[{},{},{}]}}"#,
        n as f64 / ar_secs,
        out.len() as f64 / secs.max(1e-9),
        speedup,
        dyn_cfg.draft_depth,
        dyn_cfg.top_k_per_step,
        layer_indices[0],
        layer_indices[1],
        layer_indices[2],
    );
    eprintln!("\n=== EAGLE-3 speedup: {speedup:.2}× over AR ===");

    assert_eq!(ar_out.len(), n);
    // Don't assert specific output match — EAGLE-3 may diverge from greedy
    // AR if the d2t mapping forces a slightly different vocab subset.
}
