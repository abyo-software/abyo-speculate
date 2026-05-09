//! Sampling utilities: softmax, top-p, temperature, rejection sampling.

pub mod tokens;

pub use tokens::{sample_from_distribution, softmax_with_temperature, top_p_filter};
