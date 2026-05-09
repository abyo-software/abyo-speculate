//! Vanilla Speculative Decoding (Leviathan et al. 2023).
//!
//! Algorithm sketch:
//! 1. Draft model proposes `k` tokens autoregressively.
//! 2. Target model evaluates all `k+1` positions in a single forward pass,
//!    yielding distributions `p_target[i]` for each prefix.
//! 3. For each draft token `x_i` with draft probability `q(x_i)`,
//!    accept it if `u ~ U(0,1) <= p_target(x_i) / q(x_i)`.
//! 4. On the first rejection at index `i`, sample a replacement from the
//!    *adjusted* distribution `max(0, p_target - q)` (renormalized).
//! 5. Roll target's KV cache back to position `i+1`, discard draft cache past `i`.
//!
//! The acceptance rule above provably matches the target's output distribution
//! (Leviathan §3.1 / Chen et al. 2023 lemma 1). This module's `accept_or_resample`
//! routine is the load-bearing correctness primitive — it is unit-tested with a
//! statistical KS-style check in `crates/speculate/tests/`.

#![allow(dead_code)] // Phase 1a is in flight; types will be wired up by `engine.rs`.

use crate::Result;

/// Outcome of running rejection sampling against a single draft token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceptOutcome {
    /// Draft token was accepted; continue to the next draft token.
    Accepted,
    /// Draft token was rejected; the caller must resample from the adjusted
    /// distribution and stop processing further draft tokens.
    Rejected,
}

/// Configuration for the vanilla SD generation loop.
#[derive(Debug, Clone)]
pub struct VanillaConfig {
    /// Number of tokens the draft model proposes before each verification.
    pub draft_lookahead: usize,
    /// Sampling temperature applied to *both* draft and target. `0.0` means greedy.
    pub temperature: f32,
    /// Top-p nucleus sampling threshold. `1.0` disables.
    pub top_p: f32,
}

impl Default for VanillaConfig {
    fn default() -> Self {
        Self {
            draft_lookahead: 4,
            temperature: 0.7,
            top_p: 0.95,
        }
    }
}

/// Apply Leviathan's modified-rejection rule to a single draft token.
///
/// Returns:
/// - `AcceptOutcome::Accepted` if `u <= p_target / q_draft`
/// - `AcceptOutcome::Rejected` otherwise
///
/// `q_draft` and `p_target` must both be probabilities of the *same token* under
/// the respective distributions, i.e. *not* the full distribution.
pub fn modified_rejection_step(q_draft: f32, p_target: f32, u: f32) -> AcceptOutcome {
    debug_assert!((0.0..=1.0).contains(&q_draft), "q_draft out of [0,1]");
    debug_assert!((0.0..=1.0).contains(&p_target), "p_target out of [0,1]");
    debug_assert!((0.0..=1.0).contains(&u), "u out of [0,1]");
    if q_draft <= 0.0 {
        // The draft would never have produced this token — accept (target-driven).
        return AcceptOutcome::Accepted;
    }
    let ratio = (p_target / q_draft).min(1.0);
    if u <= ratio {
        AcceptOutcome::Accepted
    } else {
        AcceptOutcome::Rejected
    }
}

/// Build the *adjusted* distribution `max(0, p_target - q_draft)` and renormalize.
///
/// Used after a rejection to sample the replacement token. Both inputs must be
/// proper distributions (sum to 1, non-negative) over the same vocabulary.
///
/// Returns `Err` if the adjusted distribution sums to zero (which can only
/// happen for numerically-pathological inputs; in practice the target almost
/// always assigns *some* probability mass that the draft did not).
pub fn adjusted_distribution(p_target: &[f32], q_draft: &[f32]) -> Result<Vec<f32>> {
    if p_target.len() != q_draft.len() {
        return Err(crate::Error::Sampling(format!(
            "vocab size mismatch: target={}, draft={}",
            p_target.len(),
            q_draft.len()
        )));
    }
    let mut adj: Vec<f32> = p_target
        .iter()
        .zip(q_draft.iter())
        .map(|(p, q)| (p - q).max(0.0))
        .collect();
    let sum: f32 = adj.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return Err(crate::Error::Sampling(format!(
            "adjusted distribution has non-positive mass: sum={sum}"
        )));
    }
    for v in adj.iter_mut() {
        *v /= sum;
    }
    Ok(adj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn always_accept_when_target_geq_draft() {
        // p_target >= q_draft → ratio >= 1 → always accept.
        for u_pct in 0..=100 {
            let u = u_pct as f32 / 100.0;
            assert_eq!(
                modified_rejection_step(0.4, 0.8, u),
                AcceptOutcome::Accepted
            );
        }
    }

    #[test]
    fn reject_when_u_above_ratio() {
        // q=0.8, p=0.4 → ratio = 0.5
        assert_eq!(
            modified_rejection_step(0.8, 0.4, 0.6),
            AcceptOutcome::Rejected
        );
        assert_eq!(
            modified_rejection_step(0.8, 0.4, 0.4),
            AcceptOutcome::Accepted
        );
    }

    #[test]
    fn adjusted_distribution_renormalizes() {
        let p = vec![0.5, 0.3, 0.2];
        let q = vec![0.1, 0.4, 0.5];
        let adj = adjusted_distribution(&p, &q).unwrap();
        // raw: (0.4, 0.0, 0.0) → sum 0.4 → (1.0, 0.0, 0.0)
        assert_relative_eq!(adj[0], 1.0, max_relative = 1e-6);
        assert_relative_eq!(adj[1], 0.0, max_relative = 1e-6);
        assert_relative_eq!(adj[2], 0.0, max_relative = 1e-6);
        let sum: f32 = adj.iter().sum();
        assert_relative_eq!(sum, 1.0, max_relative = 1e-6);
    }

    #[test]
    fn adjusted_distribution_rejects_zero_mass() {
        // p == q → all differences are 0 → no mass to sample from.
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        assert!(adjusted_distribution(&p, &q).is_err());
    }

    #[test]
    fn adjusted_distribution_rejects_size_mismatch() {
        let p = vec![0.5, 0.5];
        let q = vec![1.0];
        assert!(adjusted_distribution(&p, &q).is_err());
    }
}
