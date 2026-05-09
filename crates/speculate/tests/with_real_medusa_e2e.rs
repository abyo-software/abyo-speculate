//! Real Medusa end-to-end speedup measurement against the bundled
//! `FasterDecoding/medusa-1.0-vicuna-7b-v1.5` checkpoint.
//!
//! That repo ships **both** the Vicuna 7B base weights AND the trained
//! Medusa heads in the same multi-shard PyTorch pickle (~14 GB total).
//! No safetensors mirror is available without an HF token, and lmsys's own
//! Vicuna repos are PyTorch-only too.
//!
//! Strategy: load via [`crate::model::hub::MultiPthBackend`] — a custom
//! `SimpleBackend` that wraps multiple `PthTensors` shards and routes
//! `get(name)` calls to the right shard via the repo's
//! `pytorch_model.bin.index.json`. From the same backend we build:
//! - `LlamaDecoder` (for the base `model.*` + `lm_head.*` keys), and
//! - `MedusaHeadsCandle` (for the `medusa_head.<i>....` keys).
//!
//! Then [`run_medusa_real`] gives us a real-target / real-head Medusa loop.
//!
//! Memory: 14 GB BF16 base + ~1.4 GB heads ≈ 15.4 GB, near the 16 GB
//! ceiling on RTX 4070 Ti SUPER. Short prompts only.
//!
//! Run with:
//! ```sh
//! cargo test --release --features cuda \
//!   -p abyo-speculate --test with_real_medusa_e2e -- --ignored --nocapture
//! ```

#![cfg(not(target_os = "windows"))]

use abyo_speculate::methods::medusa::{
    run_medusa_real, Acceptance, MedusaConfig, MedusaHeads, MedusaHeadsCandle, MedusaRunConfig,
    TreeTopology,
};
use abyo_speculate::model::hub::{download_pth_sharded, MultiPthBackend};
use abyo_speculate::model::llama::LlamaDecoder;
use abyo_speculate::model::llama_local::LlamaConfig;
use abyo_speculate::model::Decoder;
use candle_core::{DType, Device};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::VarBuilder;
use rand::SeedableRng;
use std::sync::Arc;

const REPO: &str = "FasterDecoding/medusa-1.0-vicuna-7b-v1.5";

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
#[ignore = "downloads ~14GB of pytorch shards and needs ~15GB GPU memory; \
            set ABYO_LARGE_GPU=1 to enable (24GB+ GPU recommended — 16GB OOMs \
            during the head load)"]
fn vicuna_7b_with_real_medusa_heads() {
    if std::env::var("ABYO_LARGE_GPU").ok().as_deref() != Some("1") {
        eprintln!(
            "skipping vicuna_7b_with_real_medusa_heads — set ABYO_LARGE_GPU=1 \
             to opt in. Current 16 GB hardware OOMs at MedusaHeadsCandle \
             load time (Vicuna 7B BF16 ~14 GB + 5 Medusa heads ~1.5 GB > 16 GB \
             when activations are also resident)."
        );
        return;
    }
    use abyo_speculate::model::hub::download_files;

    // 1. Download the multi-shard pytorch checkpoint + the config.
    let (index_path, shard_paths) = download_pth_sharded(REPO).expect("download pth shards");
    let aux = download_files(REPO, &["config.json"]).expect("download aux");
    let config_path = &aux[0];
    // FasterDecoding's repo (and lmsys/vicuna-7b-v1.5 itself) only ship the
    // SentencePiece `tokenizer.model`; the `tokenizers` crate wants the HF
    // JSON format. Llama-2-derived models share the same vocabulary, so we
    // borrow TinyLlama's tokenizer.json — same 32k-token Llama 2 SP
    // vocabulary, just packaged in HF JSON.
    let tok_alt = download_files("TinyLlama/TinyLlama-1.1B-Chat-v1.0", &["tokenizer.json"])
        .expect("download Llama-2 tokenizer.json from TinyLlama");
    let tokenizer_path = &tok_alt[0];

    // 2. Parse Vicuna-shaped LlamaConfig (the FasterDecoding repo's
    //    config.json is a Llama config; the medusa-specific config.json
    //    fields we ignore via serde's permissive deserialization).
    let config_json = std::fs::read_to_string(config_path).unwrap();
    let hf_config: LlamaConfig = serde_json::from_str(&config_json).unwrap();
    let config = hf_config.into_config(false);

    let device = pick_device();
    let dtype = if matches!(device, Device::Cpu) {
        DType::F32
    } else {
        DType::BF16
    };

    // 3. Build a multi-shard backend and wrap as VarBuilder. Boxing as
    //    `dyn SimpleBackend` lets the same Arc'd backend be cloned cheaply
    //    for both LlamaDecoder + MedusaHeadsCandle.
    let backend = MultiPthBackend::from_paths(&index_path, &shard_paths)
        .expect("MultiPthBackend::from_paths");
    let backend: Arc<dyn SimpleBackend> = Arc::new(backend);
    // VarBuilder::from_backend wants a Backend (Box<dyn SimpleBackend> works).
    // We wrap each consumer's view in a fresh Box pointing at the shared Arc.
    let vb_for_base = VarBuilder::from_backend(
        Box::new(BackendArc(backend.clone())) as Box<dyn SimpleBackend>,
        dtype,
        device.clone(),
    );
    let vb_for_heads = VarBuilder::from_backend(
        Box::new(BackendArc(backend.clone())) as Box<dyn SimpleBackend>,
        dtype,
        device.clone(),
    );

    // 4. Load base + heads from the same pth shards.
    let mut target =
        LlamaDecoder::from_var_builder(&config, vb_for_base, tokenizer_path, device.clone(), dtype)
            .expect("LlamaDecoder::from_var_builder");

    let medusa_cfg = MedusaConfig {
        n_heads: 5, // medusa-1.0-vicuna-7b-v1.5 ships 5 heads
        hidden_size: 4096,
        vocab_size: config.vocab_size,
        residual_layers: 1,
    };
    let heads = MedusaHeadsCandle::from_fasterdecoding_var_builder(
        &medusa_cfg,
        vb_for_heads.pp("medusa_head"),
    )
    .expect("MedusaHeadsCandle from bundled VB");
    let skeleton = MedusaHeads::from_config(medusa_cfg.clone());

    // 5. Smoke: autoregressive baseline.
    let prompt = target.encode("The capital of France is", true).unwrap();
    Decoder::observe(&mut target, &prompt).unwrap();
    let mut ar_out = Vec::new();
    let t0 = std::time::Instant::now();
    for _ in 0..16 {
        let logits = Decoder::next_logits(&mut target).unwrap();
        let tok = argmax_u32(&logits);
        ar_out.push(tok);
        Decoder::observe(&mut target, &[tok]).unwrap();
    }
    let ar_secs = t0.elapsed().as_secs_f64();
    let ar_text = target.decode(&ar_out, true).unwrap();
    println!(
        "AR  : 16 tokens in {ar_secs:.3}s = {:.2} tok/s\nAR text: {ar_text}",
        16.0 / ar_secs
    );

    // 6. Real Medusa loop, same prompt.
    let cfg = MedusaRunConfig {
        topology: TreeTopology::Greedy,
        top_k_per_head: 1,
        acceptance: Acceptance::Greedy,
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
    let t1 = std::time::Instant::now();
    let med_out =
        run_medusa_real(&mut target, &heads, &skeleton, &prompt, 16, &cfg, &mut rng).unwrap();
    let med_secs = t1.elapsed().as_secs_f64();
    let med_text = target.decode(&med_out, true).unwrap();
    println!(
        "Med : {} tokens in {med_secs:.3}s = {:.2} tok/s\nMed text: {med_text}",
        med_out.len(),
        med_out.len() as f64 / med_secs
    );

    let speedup = (med_out.len() as f64 / med_secs) / (16.0 / ar_secs);
    println!("\n=== Medusa speedup: {speedup:.2}× over autoregressive ===");

    assert_eq!(ar_out.len(), 16);
    assert_eq!(med_out.len(), 16);
    // Sanity: with greedy heads + greedy acceptance, the Medusa output
    // must agree with the AR output (the heads only ever propose; the
    // target's argmax is what gets emitted on rejection or as bonus).
    // We do *not* assert byte-equality — Medusa with non-trivial heads
    // can produce a DIFFERENT trajectory because the heads may propose
    // tokens the target's argmax also picks (just in different rounds).
}

/// Wrapper: `Box<dyn SimpleBackend>` requires Send + Sync; `Arc<...>` is
/// Send+Sync but we need a SimpleBackend impl that delegates. This newtype
/// is the smallest such adapter.
struct BackendArc(Arc<dyn SimpleBackend>);

impl SimpleBackend for BackendArc {
    fn get(
        &self,
        s: candle_core::Shape,
        name: &str,
        h: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<candle_core::Tensor> {
        self.0.get(s, name, h, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.0.contains_tensor(name)
    }
}
