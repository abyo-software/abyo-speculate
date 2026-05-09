//! [`SpeculateEngine`] — the public façade that ties model loading, the chosen
//! SD method, and sampling together.
//!
//! Phase 1 ships:
//! - The builder + dispatch skeleton (this file)
//! - A working `Method::Autoregressive` path that does real generation with a
//!   loaded [`Decoder`]
//! - A `Method::Vanilla` path that runs the
//!   [reference SD loop](crate::methods::vanilla::run_vanilla_sd) against a
//!   loaded target + draft pair
//!
//! Medusa / EAGLE land in Phase 1b / 2.

use crate::{
    methods::Method,
    model::{loader::ModelSource, Decoder},
    sampling::{sample_from_distribution, softmax_with_temperature, top_p_filter},
    Error, Result,
};

/// Top-level entry point.
///
/// `SpeculateEngine` is split into a *config-only* form (returned by
/// [`SpeculateEngine::builder`]) and a *live* form that owns loaded models.
/// Use [`SpeculateEngine::with_target`] to attach a [`Decoder`] (and
/// [`SpeculateEngine::with_draft`] for the optional draft model).
pub struct SpeculateEngine {
    config: EngineConfig,
    target: Option<Box<dyn Decoder + Send>>,
    draft: Option<Box<dyn Decoder + Send>>,
}

impl std::fmt::Debug for SpeculateEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeculateEngine")
            .field("config", &self.config)
            .field("target_loaded", &self.target.is_some())
            .field("draft_loaded", &self.draft.is_some())
            .finish()
    }
}

/// Resolved configuration for a fully-built engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Target model source.
    pub target: ModelSource,
    /// Optional draft model source. Required iff [`Method::needs_draft_model`].
    pub draft: Option<ModelSource>,
    /// SD method.
    pub method: Method,
    /// Maximum number of tokens to generate per `generate()` call.
    pub default_max_tokens: usize,
    /// Sampling temperature.
    pub temperature: f32,
    /// Top-p nucleus threshold (1.0 disables).
    pub top_p: f32,
    /// Number of tokens the draft proposes per verification round.
    pub draft_lookahead: usize,
    /// RNG seed (deterministic generation when set).
    pub seed: Option<u64>,
}

impl SpeculateEngine {
    /// Begin configuring an engine.
    pub fn builder() -> SpeculateEngineBuilder {
        SpeculateEngineBuilder::default()
    }

    /// Construct an engine from a preset model name. Returns a config-only
    /// engine; the caller is responsible for loading the actual model via
    /// [`SpeculateEngine::with_target`] (and `with_draft` if applicable).
    pub fn preset_for(model_name: &str) -> Result<Self> {
        let preset = crate::presets::lookup(model_name)
            .ok_or_else(|| Error::UnknownPreset(model_name.to_string()))?;
        SpeculateEngineBuilder::default()
            .target_model(&preset.target)
            .method(preset.method)
            .maybe_draft_model(preset.draft.as_deref())
            .draft_lookahead(preset.draft_lookahead)
            .temperature(preset.temperature)
            .top_p(preset.top_p)
            .build()
    }

    /// Read-only view on the resolved configuration.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Attach a loaded target [`Decoder`] (e.g. [`crate::model::qwen2::Qwen2Decoder`]).
    /// Returns `self` for chaining.
    pub fn with_target<D: Decoder + Send + 'static>(mut self, target: D) -> Self {
        self.target = Some(Box::new(target));
        self
    }

    /// Attach a loaded draft [`Decoder`].
    pub fn with_draft<D: Decoder + Send + 'static>(mut self, draft: D) -> Self {
        self.draft = Some(Box::new(draft));
        self
    }

    /// Whether the engine has all required models loaded for its configured method.
    pub fn is_ready(&self) -> bool {
        if self.target.is_none() {
            return false;
        }
        if self.config.method.needs_draft_model() && self.draft.is_none() {
            return false;
        }
        true
    }

    /// Generate `max_new_tokens` from a token-id prompt.
    ///
    /// Lower-level than [`Self::generate`]; useful when you already have a
    /// tokenized prompt or want to keep tokenization out of this crate.
    pub fn generate_tokens(&mut self, prompt: &[u32], max_new_tokens: usize) -> Result<Vec<u32>> {
        if !self.is_ready() {
            return Err(Error::MissingField(
                "models not loaded — call with_target / with_draft first",
            ));
        }
        let mut rng: Box<dyn rand::RngCore> = match self.config.seed {
            Some(s) => {
                use rand::SeedableRng;
                Box::new(rand::rngs::StdRng::seed_from_u64(s))
            }
            None => Box::new(rand::thread_rng()),
        };
        match self.config.method {
            Method::Autoregressive => {
                let target = self.target.as_mut().unwrap();
                run_autoregressive(
                    target.as_mut(),
                    prompt,
                    max_new_tokens,
                    &self.config,
                    &mut rng,
                )
            }
            Method::Vanilla => {
                let target = self.target.as_mut().unwrap();
                let draft = self.draft.as_mut().unwrap();
                let cfg = crate::methods::vanilla::VanillaConfig {
                    draft_lookahead: self.config.draft_lookahead,
                    temperature: self.config.temperature,
                    top_p: self.config.top_p,
                };
                crate::methods::vanilla::run_vanilla_sd(
                    target.as_mut(),
                    draft.as_mut(),
                    prompt,
                    max_new_tokens,
                    &cfg,
                    &mut rng,
                )
            }
            other => Err(Error::UnsupportedMethod {
                method: other.name(),
                reason: "method not yet implemented in Phase 1".into(),
            }),
        }
    }

    /// Friendly text-in / text-out wrapper. Returns `Err` if no target model
    /// has been attached, or if the target model has no associated tokenizer
    /// (custom decoders without tokenization can use [`Self::generate_tokens`]
    /// directly).
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        let _ = (prompt, max_tokens);
        // The Decoder trait is intentionally tokenizer-agnostic. Tokenization
        // belongs to the concrete decoder (e.g. Qwen2Decoder::encode/decode).
        // For now we expose a clear error directing callers to the lower-level
        // path — wrapping tokenization here would require either coupling
        // Decoder to tokenizers or duplicating the API.
        Err(Error::Other(anyhow::anyhow!(
            "engine.generate(text) requires a tokenizer-aware backend; \
             use the model's encode + engine.generate_tokens + decode directly. \
             A higher-level wrapper lands when the Backend trait stabilizes."
        )))
    }
}

/// Plain autoregressive generation: sample from `target.next_logits` one token
/// at a time. Used as the `Method::Autoregressive` baseline.
fn run_autoregressive<R: rand::Rng>(
    target: &mut dyn Decoder,
    prompt: &[u32],
    max_new_tokens: usize,
    config: &EngineConfig,
    rng: &mut R,
) -> Result<Vec<u32>> {
    target.reset();
    target.observe(prompt)?;
    let mut out = Vec::with_capacity(max_new_tokens);
    for _ in 0..max_new_tokens {
        let logits = target.next_logits()?;
        let mut probs = softmax_with_temperature(&logits, config.temperature)?;
        if config.top_p < 1.0 {
            top_p_filter(&mut probs, config.top_p)?;
        }
        let tok = sample_from_distribution(rng, &probs)? as u32;
        target.observe(&[tok])?;
        out.push(tok);
    }
    Ok(out)
}

/// Builder for [`SpeculateEngine`].
#[derive(Debug, Default, Clone)]
pub struct SpeculateEngineBuilder {
    target: Option<ModelSource>,
    draft: Option<ModelSource>,
    method: Option<Method>,
    default_max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    draft_lookahead: Option<usize>,
    seed: Option<u64>,
}

impl SpeculateEngineBuilder {
    /// Set the target model (the one whose distribution we want to match).
    pub fn target_model(mut self, source: &str) -> Self {
        self.target = Some(ModelSource::parse(source));
        self
    }

    /// Set the draft model (small / fast, used for speculation).
    pub fn draft_model(mut self, source: &str) -> Self {
        self.draft = Some(ModelSource::parse(source));
        self
    }

    /// Set the draft model only if `Some`.
    pub fn maybe_draft_model(mut self, source: Option<&str>) -> Self {
        if let Some(s) = source {
            self.draft = Some(ModelSource::parse(s));
        }
        self
    }

    /// Set the path to the draft head / draft model (alias for [`Self::draft_model`]).
    pub fn draft_path(self, source: &str) -> Self {
        self.draft_model(source)
    }

    /// Set the SD method.
    pub fn method(mut self, m: Method) -> Self {
        self.method = Some(m);
        self
    }

    /// Default `max_tokens` if not overridden in [`SpeculateEngine::generate`].
    pub fn default_max_tokens(mut self, n: usize) -> Self {
        self.default_max_tokens = Some(n);
        self
    }

    /// Sampling temperature.
    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Top-p nucleus threshold.
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Number of tokens the draft proposes per verification round.
    pub fn draft_lookahead(mut self, n: usize) -> Self {
        self.draft_lookahead = Some(n);
        self
    }

    /// Set a deterministic RNG seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    /// Validate inputs and build the engine (config-only; attach models with
    /// [`SpeculateEngine::with_target`] / `with_draft`).
    pub fn build(self) -> Result<SpeculateEngine> {
        let target = self.target.ok_or(Error::MissingField("target_model"))?;
        let method = self.method.unwrap_or(Method::Autoregressive);

        if method.needs_draft_model() && self.draft.is_none() {
            return Err(Error::UnsupportedMethod {
                method: method.name(),
                reason: "method requires a draft model; call .draft_model(...)".into(),
            });
        }

        let config = EngineConfig {
            target,
            draft: self.draft,
            method,
            default_max_tokens: self.default_max_tokens.unwrap_or(256),
            temperature: self.temperature.unwrap_or(0.7),
            top_p: self.top_p.unwrap_or(0.95),
            draft_lookahead: self.draft_lookahead.unwrap_or(4),
            seed: self.seed,
        };
        Ok(SpeculateEngine {
            config,
            target: None,
            draft: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::mock::fixed_distribution;

    #[test]
    fn builder_requires_target() {
        let err = SpeculateEngineBuilder::default().build().unwrap_err();
        assert!(matches!(err, Error::MissingField(_)));
    }

    #[test]
    fn vanilla_method_requires_draft_in_config() {
        let err = SpeculateEngineBuilder::default()
            .target_model("meta-llama/Llama-3.1-8B-Instruct")
            .method(Method::Vanilla)
            .build()
            .unwrap_err();
        assert!(matches!(err, Error::UnsupportedMethod { .. }));
    }

    #[test]
    fn autoregressive_does_not_require_draft() {
        let engine = SpeculateEngineBuilder::default()
            .target_model("meta-llama/Llama-3.1-8B-Instruct")
            .method(Method::Autoregressive)
            .build()
            .unwrap();
        assert_eq!(engine.config().method, Method::Autoregressive);
        assert!(engine.config().draft.is_none());
        assert!(!engine.is_ready(), "no model attached yet");
    }

    #[test]
    fn vanilla_with_draft_succeeds() {
        let engine = SpeculateEngineBuilder::default()
            .target_model("meta-llama/Llama-3.1-8B-Instruct")
            .draft_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            .method(Method::Vanilla)
            .draft_lookahead(6)
            .build()
            .unwrap();
        assert_eq!(engine.config().draft_lookahead, 6);
        assert!(engine.config().draft.is_some());
    }

    #[test]
    fn generate_tokens_runs_autoregressive_with_mock() {
        // Wire a fixed-distribution mock as the target and verify we get the
        // requested number of tokens back, all from the supported support.
        let target = fixed_distribution(vec![0.5, 0.3, 0.2]);
        let mut engine = SpeculateEngineBuilder::default()
            .target_model("dummy")
            .method(Method::Autoregressive)
            .seed(42)
            .build()
            .unwrap()
            .with_target(target);
        assert!(engine.is_ready());

        let out = engine.generate_tokens(&[7u32], 8).unwrap();
        assert_eq!(out.len(), 8);
        for &t in &out {
            assert!(t < 3, "produced token {t} outside vocab");
        }
    }

    #[test]
    fn generate_tokens_runs_vanilla_sd_with_mocks() {
        let target = fixed_distribution(vec![0.6, 0.3, 0.1]);
        let draft = fixed_distribution(vec![0.33, 0.33, 0.34]);
        let mut engine = SpeculateEngineBuilder::default()
            .target_model("dummy-target")
            .draft_model("dummy-draft")
            .method(Method::Vanilla)
            .draft_lookahead(3)
            .seed(99)
            .build()
            .unwrap()
            .with_target(target)
            .with_draft(draft);
        assert!(engine.is_ready());

        let out = engine.generate_tokens(&[1u32], 12).unwrap();
        assert_eq!(out.len(), 12);
    }

    #[test]
    fn generate_text_is_explicit_about_unsupported_path() {
        let target = fixed_distribution(vec![0.5, 0.5]);
        let mut engine = SpeculateEngineBuilder::default()
            .target_model("dummy")
            .build()
            .unwrap()
            .with_target(target);
        let err = engine.generate("hi", 5).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("Backend") || msg.contains("tokenizer"),
            "expected guidance toward the lower-level path; got: {msg}"
        );
    }
}
