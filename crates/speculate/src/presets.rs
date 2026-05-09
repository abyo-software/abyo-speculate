//! Curated configurations for the four supported model families.
//!
//! Each preset bundles target / draft model identifiers, the recommended SD
//! method, and tuned sampling defaults. Presets evolve as we benchmark — if
//! you depend on exact numerical behavior, pin a crate version.

use crate::methods::Method;

/// One preset configuration for a known model.
#[derive(Debug, Clone)]
pub struct Preset {
    /// Hugging Face id (or path) for the target model.
    pub target: String,
    /// Optional draft / Medusa-head id.
    pub draft: Option<String>,
    /// Recommended SD method.
    pub method: Method,
    /// Default temperature.
    pub temperature: f32,
    /// Default top-p.
    pub top_p: f32,
    /// Default draft lookahead `k`.
    pub draft_lookahead: usize,
}

/// Look up a preset by short name (e.g. `"llama-3.1-8b"`).
pub fn lookup(name: &str) -> Option<Preset> {
    let n = name.trim().to_lowercase();
    Some(match n.as_str() {
        "llama-3.1-8b" | "llama-3.1-8b-instruct" => Preset {
            target: "meta-llama/Llama-3.1-8B-Instruct".into(),
            // EAGLE-2 head availability TBD — wired in Phase 2b once we confirm HF distro.
            draft: Some("TinyLlama/TinyLlama-1.1B-Chat-v1.0".into()),
            method: Method::Vanilla,
            temperature: 0.7,
            top_p: 0.9,
            draft_lookahead: 4,
        },
        "qwen-2.5-7b" | "qwen-2.5-7b-instruct" => Preset {
            target: "Qwen/Qwen2.5-7B-Instruct".into(),
            draft: Some("Qwen/Qwen2.5-0.5B-Instruct".into()),
            method: Method::Vanilla,
            temperature: 0.7,
            top_p: 0.9,
            draft_lookahead: 5,
        },
        "mistral-7b" | "mistral-7b-instruct" => Preset {
            target: "mistralai/Mistral-7B-Instruct-v0.3".into(),
            draft: None,
            method: Method::Medusa,
            temperature: 0.7,
            top_p: 0.9,
            draft_lookahead: 4,
        },
        "phi-3.5-mini" | "phi-3.5" => Preset {
            target: "microsoft/Phi-3.5-mini-instruct".into(),
            draft: None,
            method: Method::Medusa,
            temperature: 0.7,
            top_p: 0.9,
            draft_lookahead: 4,
        },
        _ => return None,
    })
}

/// Names of the presets shipped with this crate (for CLI listing).
pub fn known_names() -> &'static [&'static str] {
    &["llama-3.1-8b", "qwen-2.5-7b", "mistral-7b", "phi-3.5-mini"]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_names_are_resolvable() {
        for &n in known_names() {
            assert!(lookup(n).is_some(), "preset {n} should resolve");
        }
    }

    #[test]
    fn unknown_name_returns_none() {
        assert!(lookup("totally-not-a-model").is_none());
    }

    #[test]
    fn case_and_alias_normalization() {
        assert!(lookup("Llama-3.1-8B").is_some());
        assert!(lookup("llama-3.1-8b-instruct").is_some());
    }
}
