//! Rollback-capable KV cache.
//!
//! The cache is keyed per layer × per head; we store contiguous `Tensor`s along
//! the sequence dimension and track the *committed length* (last position the
//! target has accepted). Snapshots record only the committed length — the
//! underlying tensor storage is reused, so rollback is O(1).
//!
//! The current implementation is intentionally simple:
//! - Only one outstanding snapshot at a time (caller must drop / commit).
//! - No support for concurrent rollback paths (Medusa tree-style verification
//!   is handled at a higher layer using a different cache shape).

use crate::{Error, Result};
use candle_core::{Device, Tensor};

/// Per-layer rollback-capable KV cache.
///
/// Conceptually:
/// ```text
///   committed: [-------- pos 0..C --------]   ← always trusted
///   tentative: [----------------- pos 0..T -----------------]
/// ```
/// where `C <= T`. Rollback truncates `T` back to `C`. Commit promotes `T → C`.
#[derive(Debug)]
pub struct RollbackCache {
    /// Cached keys, shape `[batch, n_heads, seq_len_total, head_dim]`.
    keys: Option<Tensor>,
    /// Cached values, same shape as `keys`.
    values: Option<Tensor>,
    /// Number of positions the *target* has accepted (the safe rollback point).
    committed_len: usize,
    /// Total number of positions currently in the cache (committed + tentative).
    total_len: usize,
    device: Device,
}

/// Lightweight handle returned by [`RollbackCache::snapshot`]; pass to
/// [`RollbackCache::rollback`] to discard tentative entries appended since.
#[derive(Debug, Clone, Copy)]
pub struct KvSnapshot {
    committed_len: usize,
}

impl RollbackCache {
    /// Create an empty cache bound to `device`.
    pub fn new(device: Device) -> Self {
        Self {
            keys: None,
            values: None,
            committed_len: 0,
            total_len: 0,
            device,
        }
    }

    /// Number of positions in the cache (committed + tentative).
    pub fn total_len(&self) -> usize {
        self.total_len
    }

    /// Number of positions the target has accepted.
    pub fn committed_len(&self) -> usize {
        self.committed_len
    }

    /// Capture the current commit position. Cheap; safe to call any time.
    pub fn snapshot(&self) -> KvSnapshot {
        KvSnapshot {
            committed_len: self.committed_len,
        }
    }

    /// Append `(k, v)` to the *tentative* portion of the cache.
    ///
    /// `k`, `v` must share shape `[batch, n_heads, n_new, head_dim]`.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let new_len = k.dim(2)?;
        debug_assert_eq!(new_len, v.dim(2)?, "append: k seq_len must equal v seq_len");

        match (&self.keys, &self.values) {
            (None, None) => {
                self.keys = Some(k.clone());
                self.values = Some(v.clone());
            }
            (Some(prev_k), Some(prev_v)) => {
                let trimmed_k = prev_k.narrow(2, 0, self.total_len)?;
                let trimmed_v = prev_v.narrow(2, 0, self.total_len)?;
                self.keys = Some(Tensor::cat(&[&trimmed_k, k], 2)?);
                self.values = Some(Tensor::cat(&[&trimmed_v, v], 2)?);
            }
            _ => unreachable!("keys/values invariant: both Some or both None"),
        }
        self.total_len += new_len;
        Ok(())
    }

    /// Promote tentative entries into committed entries (no-op on shape).
    pub fn commit(&mut self) {
        self.committed_len = self.total_len;
    }

    /// Restore the cache to the state captured by `snap`.
    ///
    /// Returns an error if `snap` predates the current commit point — that would
    /// imply someone is trying to roll back *into* trusted committed territory,
    /// which is always a bug in the caller.
    pub fn rollback(&mut self, snap: KvSnapshot) -> Result<()> {
        if snap.committed_len > self.committed_len {
            return Err(Error::CacheRollback(format!(
                "snapshot points to length {}, but cache has only committed {}",
                snap.committed_len, self.committed_len
            )));
        }
        self.total_len = snap.committed_len;
        self.committed_len = snap.committed_len;
        Ok(())
    }

    /// View on the cache trimmed to the currently-active (tentative) length.
    pub fn current(&self) -> Result<Option<(Tensor, Tensor)>> {
        match (&self.keys, &self.values) {
            (Some(k), Some(v)) => {
                let k = k.narrow(2, 0, self.total_len)?;
                let v = v.narrow(2, 0, self.total_len)?;
                Ok(Some((k, v)))
            }
            (None, None) => Ok(None),
            _ => unreachable!(),
        }
    }

    /// Device the underlying tensors live on.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn fake_kv(seq: usize, device: &Device) -> (Tensor, Tensor) {
        // [batch=1, heads=2, seq, head_dim=4]
        let shape = (1usize, 2usize, seq, 4usize);
        let k = Tensor::ones(shape, DType::F32, device).unwrap();
        let v = Tensor::ones(shape, DType::F32, device).unwrap();
        (k, v)
    }

    #[test]
    fn append_extends_total_len() {
        let dev = Device::Cpu;
        let mut c = RollbackCache::new(dev.clone());
        let (k, v) = fake_kv(3, &dev);
        c.append(&k, &v).unwrap();
        assert_eq!(c.total_len(), 3);
        assert_eq!(c.committed_len(), 0);
    }

    #[test]
    fn commit_advances_committed_len() {
        let dev = Device::Cpu;
        let mut c = RollbackCache::new(dev.clone());
        let (k, v) = fake_kv(3, &dev);
        c.append(&k, &v).unwrap();
        c.commit();
        assert_eq!(c.committed_len(), 3);
        assert_eq!(c.total_len(), 3);
    }

    #[test]
    fn rollback_truncates_to_snapshot() {
        let dev = Device::Cpu;
        let mut c = RollbackCache::new(dev.clone());
        let (k1, v1) = fake_kv(3, &dev);
        c.append(&k1, &v1).unwrap();
        c.commit();
        let snap = c.snapshot();

        // Append tentative and roll it back.
        let (k2, v2) = fake_kv(5, &dev);
        c.append(&k2, &v2).unwrap();
        assert_eq!(c.total_len(), 8);
        c.rollback(snap).unwrap();
        assert_eq!(c.total_len(), 3);
        assert_eq!(c.committed_len(), 3);
    }

    #[test]
    fn rollback_to_future_snapshot_is_error() {
        let dev = Device::Cpu;
        let mut c = RollbackCache::new(dev.clone());
        // Snapshot at len 0, then commit some, then try to "rollback" to a fabricated future.
        let bogus = KvSnapshot { committed_len: 99 };
        let (k, v) = fake_kv(2, &dev);
        c.append(&k, &v).unwrap();
        c.commit();
        assert!(c.rollback(bogus).is_err());
    }
}
