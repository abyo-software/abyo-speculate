//! KV-cache primitives with rollback support.
//!
//! Speculative decoding requires the target model to *evaluate* `k` draft tokens
//! at once, then *roll back* its KV cache to wherever the first rejection
//! occurred. candle's stock KV-cache implementation (`candle_nn::kv_cache`)
//! supports append-and-truncate, but the SD-specific snapshot/restore lifecycle
//! is wrapped here so callers don't have to think about it.

pub mod rollback;

pub use rollback::{KvSnapshot, RollbackCache};
