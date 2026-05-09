//! SD method implementations.
//!
//! Each method exposes the same `Decoder` trait so the [`crate::SpeculateEngine`]
//! can dispatch generically.

pub mod eagle;
pub mod medusa;
pub mod vanilla;

/// Identifier for which Speculative Decoding algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Method {
    /// Plain autoregressive — no speculation, useful as a baseline.
    Autoregressive,
    /// Leviathan et al. 2023 — separate draft model + rejection sampling.
    Vanilla,
    /// Cai et al. 2024 — multiple decoding heads, no separate draft model.
    Medusa,
    /// Li et al. 2024 — feature-level draft + dynamic tree.
    Eagle2,
    /// Li et al. 2025 — multi-layer feature draft.
    Eagle3,
}

impl Method {
    /// Human-readable name (matches the published paper convention).
    pub const fn name(self) -> &'static str {
        match self {
            Method::Autoregressive => "autoregressive",
            Method::Vanilla => "vanilla-sd",
            Method::Medusa => "medusa",
            Method::Eagle2 => "eagle-2",
            Method::Eagle3 => "eagle-3",
        }
    }

    /// Whether this method requires a separate draft model checkpoint.
    pub const fn needs_draft_model(self) -> bool {
        matches!(self, Method::Vanilla | Method::Eagle2 | Method::Eagle3)
    }
}
