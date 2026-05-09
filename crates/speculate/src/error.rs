//! Crate error types.

use thiserror::Error;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// All errors that abyo-speculate can return.
#[derive(Debug, Error)]
pub enum Error {
    /// A required builder field was not set.
    #[error("missing required builder field: {0}")]
    MissingField(&'static str),

    /// The requested model is not in the recognized preset list.
    #[error("unknown model preset: {0}")]
    UnknownPreset(String),

    /// The requested SD method does not support the requested operation in this configuration.
    #[error("method {method} unsupported here: {reason}")]
    UnsupportedMethod {
        /// The method name (e.g. "Medusa").
        method: &'static str,
        /// Why it's unsupported in this context.
        reason: String,
    },

    /// A KV-cache snapshot could not be restored (rollback misuse).
    #[error("invalid KV-cache rollback: {0}")]
    CacheRollback(String),

    /// Sampling produced an invalid distribution (NaN, all-zero, etc).
    #[error("invalid sampling distribution: {0}")]
    Sampling(String),

    /// Pass-through for [`candle_core::Error`].
    #[error(transparent)]
    Candle(#[from] candle_core::Error),

    /// Pass-through for tokenizer errors.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// Pass-through for model-load / IO errors.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Anything not yet specifically classified.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl From<tokenizers::Error> for Error {
    fn from(e: tokenizers::Error) -> Self {
        Error::Tokenizer(e.to_string())
    }
}
