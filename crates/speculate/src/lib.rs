//! # abyo-speculate
//!
//! Pure Rust Speculative Decoding library for local LLMs, optimized for batch size 1.
//!
//! See the [crate README](https://github.com/abyo-software/abyo-speculate) and
//! the project plan for design context.
//!
//! ## Quick example
//!
//! ```no_run
//! use abyo_speculate::{SpeculateEngine, Method};
//!
//! # fn main() -> anyhow::Result<()> {
//! let mut engine = SpeculateEngine::builder()
//!     .target_model("llama-3.1-8b-instruct")
//!     .method(Method::Vanilla)
//!     .draft_model("tinyllama-1.1b")
//!     .build()?;
//!
//! // engine.with_target(...).with_draft(...) attach loaded models;
//! // see model::qwen2::Qwen2Decoder for a concrete loader.
//! let _tokens = engine.generate_tokens(&[1u32, 2, 3], 64)?;
//! # Ok(())
//! # }
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod cache;
pub mod device;
pub mod engine;
pub mod error;
pub mod methods;
pub mod model;
pub mod presets;
pub mod sampling;
pub mod tree;

pub use engine::{SpeculateEngine, SpeculateEngineBuilder};
pub use error::{Error, Result};
pub use methods::Method;
