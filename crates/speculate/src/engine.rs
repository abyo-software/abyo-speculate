//! [`SpeculateEngine`] — the public façade that ties model loading, the chosen
//! SD method, and sampling together.
//!
//! Phase 1 ships the **builder + dispatch skeleton** with a working
//! `Method::Autoregressive` path that does plain greedy decoding with no
//! speculation. This gives downstream work (correctness tests, bench harness,
//! preset wiring) a stable interface to depend on while the SD methods land.

use crate::{methods::Method, model::loader::ModelSource, Error, Result};

/// Top-level entry point.
#[derive(Debug)]
pub struct SpeculateEngine {
    config: EngineConfig,
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
}

impl SpeculateEngine {
    /// Begin configuring an engine.
    pub fn builder() -> SpeculateEngineBuilder {
        SpeculateEngineBuilder::default()
    }

    /// Construct an engine from a preset model name.
    ///
    /// See [`crate::presets`] for the catalog.
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

    /// Generate `max_tokens` continuation tokens from `prompt`.
    ///
    /// Phase 1: returns a friendly placeholder that *does not* actually run the
    /// model. The wiring to candle lands in the Phase-1a milestone; this stub
    /// exists so downstream crates and the CLI can compile and integrate against
    /// the API surface today.
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let _ = max_tokens;
        Ok(format!(
            "[abyo-speculate v0.0.1: model wiring not yet implemented; method={}, prompt_len={}]",
            self.config.method.name(),
            prompt.len()
        ))
    }
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

    /// Validate inputs and build the engine.
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
        };
        Ok(SpeculateEngine { config })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_requires_target() {
        let err = SpeculateEngineBuilder::default().build().unwrap_err();
        assert!(matches!(err, Error::MissingField("target_model")));
    }

    #[test]
    fn vanilla_method_requires_draft() {
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
    fn generate_returns_placeholder_for_now() {
        let engine = SpeculateEngineBuilder::default()
            .target_model("meta-llama/Llama-3.1-8B-Instruct")
            .build()
            .unwrap();
        let out = engine.generate("hi", 10).unwrap();
        assert!(out.contains("abyo-speculate"));
    }
}
