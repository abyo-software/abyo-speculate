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

/// Run one full vanilla SD generation loop using a [`crate::model::mock::MockDecoder`]
/// pair. This is the **correctness reference implementation**: every other SD
/// path in the crate is unit-tested for distribution-matching against this loop
/// (which is itself unit-tested against analytic distributions).
///
/// Algorithm: Leviathan et al. 2023, with the modified rejection rule
/// `accept iff u <= min(1, p_target(x) / q_draft(x))` and a re-sample on
/// rejection from `norm(max(0, p_target - q_draft))`.
#[cfg(any(test, feature = "test-util"))]
pub fn run_vanilla_sd_with_mock<R: rand::Rng>(
    target: &mut crate::model::mock::MockDecoder,
    draft: &mut crate::model::mock::MockDecoder,
    prompt: &[u32],
    max_new_tokens: usize,
    config: &VanillaConfig,
    rng: &mut R,
) -> Result<Vec<u32>> {
    use crate::sampling::{sample_from_distribution, softmax_with_temperature};

    target.reset();
    draft.reset();
    target.observe(prompt);
    draft.observe(prompt);

    let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);

    while generated.len() < max_new_tokens {
        let remaining = max_new_tokens - generated.len();
        let k = config.draft_lookahead.min(remaining);
        if k == 0 {
            break;
        }

        // 1. Draft k tokens autoregressively from the draft model.
        let mut draft_tokens: Vec<u32> = Vec::with_capacity(k);
        let mut draft_dists: Vec<Vec<f32>> = Vec::with_capacity(k);
        for _ in 0..k {
            let logits = draft.next_logits();
            let probs = softmax_with_temperature(&logits, config.temperature)?;
            let tok = sample_from_distribution(rng, &probs)? as u32;
            draft_tokens.push(tok);
            draft_dists.push(probs);
            draft.observe(&[tok]);
        }

        // 2. Target evaluates [committed, draft_1, ..., draft_k] in parallel.
        // batched_logits returns k+1 logit vectors: one per prefix.
        let target_batched = target.batched_logits(&draft_tokens);
        debug_assert_eq!(target_batched.len(), k + 1);

        // 3. Walk through each draft position, accept/reject.
        let mut accepted_count = 0usize;
        let mut rejected = false;
        for i in 0..k {
            let p_probs = softmax_with_temperature(&target_batched[i], config.temperature)?;
            let q_probs = &draft_dists[i];
            let token = draft_tokens[i] as usize;
            let p = p_probs[token];
            let q = q_probs[token];
            let u: f32 = rng.gen();
            match modified_rejection_step(q, p, u) {
                AcceptOutcome::Accepted => {
                    accepted_count += 1;
                    target.observe(&[draft_tokens[i]]);
                    generated.push(draft_tokens[i]);
                    if generated.len() >= max_new_tokens {
                        break;
                    }
                }
                AcceptOutcome::Rejected => {
                    let adj = adjusted_distribution(&p_probs, q_probs)?;
                    let new_tok = sample_from_distribution(rng, &adj)? as u32;
                    target.observe(&[new_tok]);
                    generated.push(new_tok);
                    rejected = true;
                    break;
                }
            }
        }

        // 4. If all k draft tokens accepted, sample the bonus token from
        //    target_batched[k] (the distribution after the last accepted draft).
        if !rejected && accepted_count == k && generated.len() < max_new_tokens {
            let p_probs = softmax_with_temperature(&target_batched[k], config.temperature)?;
            let tok = sample_from_distribution(rng, &p_probs)? as u32;
            target.observe(&[tok]);
            generated.push(tok);
        }

        // 5. Re-sync draft history to match target. Wasteful (full re-observe),
        //    but correctness-first. The real-model path uses KV-cache rollback.
        let synced: Vec<u32> = target.history().to_vec();
        draft.reset();
        draft.observe(&synced);
    }

    Ok(generated)
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

    // ======================================================================
    // Statistical correctness tests: SD must produce samples from the *target*
    // distribution, regardless of how the draft is mis-specified.
    // ======================================================================

    use crate::model::mock::fixed_distribution;
    use rand::SeedableRng;

    /// Total-variation distance between two categorical distributions.
    fn tv_distance(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        0.5 * a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum::<f32>()
    }

    /// Run SD `trials` times with `max_new_tokens = 1`, collect the empirical
    /// distribution of the first generated token, return it.
    fn empirical_first_token(
        target_probs: Vec<f32>,
        draft_probs: Vec<f32>,
        trials: usize,
        seed: u64,
        config: &VanillaConfig,
    ) -> Vec<f32> {
        let vocab = target_probs.len();
        let mut counts = vec![0u64; vocab];
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        for _ in 0..trials {
            let mut target = fixed_distribution(target_probs.clone());
            let mut draft = fixed_distribution(draft_probs.clone());
            let prompt = [0u32]; // arbitrary, irrelevant for fixed-distribution mocks
            let out =
                run_vanilla_sd_with_mock(&mut target, &mut draft, &prompt, 1, config, &mut rng)
                    .unwrap();
            assert_eq!(out.len(), 1);
            counts[out[0] as usize] += 1;
        }
        counts
            .iter()
            .map(|&c| c as f32 / trials as f32)
            .collect::<Vec<_>>()
    }

    #[test]
    fn sd_matches_target_when_draft_is_uniform() {
        // Target is skewed; draft is uniform — worst-case for acceptance rate
        // but the *output distribution* must still match target.
        let target = vec![0.5, 0.3, 0.15, 0.05];
        let draft = vec![0.25, 0.25, 0.25, 0.25];
        let cfg = VanillaConfig {
            draft_lookahead: 4,
            temperature: 1.0,
            top_p: 1.0,
        };
        let trials = 20_000;
        let empirical = empirical_first_token(target.clone(), draft, trials, 11, &cfg);
        let tv = tv_distance(&empirical, &target);
        // 20k samples, 4-token vocab → SE per bin ~0.003. TV typically ~0.005-0.015.
        assert!(
            tv < 0.025,
            "TV distance {tv} too large; empirical={empirical:?}, target={target:?}"
        );
    }

    #[test]
    fn sd_matches_target_when_draft_is_skewed_opposite() {
        // Draft prefers token 3, target prefers token 0 — strongest mismatch.
        let target = vec![0.7, 0.15, 0.1, 0.05];
        let draft = vec![0.05, 0.1, 0.15, 0.7];
        let cfg = VanillaConfig {
            draft_lookahead: 4,
            temperature: 1.0,
            top_p: 1.0,
        };
        let trials = 20_000;
        let empirical = empirical_first_token(target.clone(), draft, trials, 23, &cfg);
        let tv = tv_distance(&empirical, &target);
        assert!(
            tv < 0.03,
            "TV distance {tv} too large; empirical={empirical:?}, target={target:?}"
        );
    }

    #[test]
    fn sd_matches_target_when_draft_equals_target() {
        // Identical distributions → acceptance rate should be 100%, but the
        // empirical output is sampled from target either way. Mostly a
        // sanity check that the loop terminates.
        let target = vec![0.4, 0.3, 0.2, 0.1];
        let cfg = VanillaConfig {
            draft_lookahead: 4,
            temperature: 1.0,
            top_p: 1.0,
        };
        let trials = 10_000;
        let empirical = empirical_first_token(target.clone(), target.clone(), trials, 7, &cfg);
        let tv = tv_distance(&empirical, &target);
        assert!(
            tv < 0.025,
            "TV distance {tv}; empirical={empirical:?}, target={target:?}"
        );
    }

    #[test]
    fn sd_emits_only_supported_target_tokens() {
        // Target gives 0 mass to tokens 2 and 3, draft is uniform. Output should
        // *never* contain 2 or 3 (the modified-rejection rule guarantees this:
        // p_target(x) = 0 → ratio = 0 → rejected; resampled from
        // max(0, p_target - q_draft) which has 0 mass on 2,3).
        let target = vec![0.6, 0.4, 0.0, 0.0];
        let draft = vec![0.25, 0.25, 0.25, 0.25];
        let cfg = VanillaConfig {
            draft_lookahead: 4,
            temperature: 1.0,
            top_p: 1.0,
        };
        let mut target_dec = fixed_distribution(target.clone());
        let mut draft_dec = fixed_distribution(draft);
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        for _ in 0..2_000 {
            let out = run_vanilla_sd_with_mock(
                &mut target_dec,
                &mut draft_dec,
                &[0u32],
                1,
                &cfg,
                &mut rng,
            )
            .unwrap();
            assert!(
                (out[0] as usize) < 2,
                "produced unsupported token {} from a target with zero mass on it",
                out[0]
            );
        }
    }

    #[test]
    fn sd_produces_max_new_tokens_count() {
        let target = vec![0.4, 0.3, 0.2, 0.1];
        let draft = vec![0.25, 0.25, 0.25, 0.25];
        let cfg = VanillaConfig {
            draft_lookahead: 3,
            temperature: 1.0,
            top_p: 1.0,
        };
        let mut t = fixed_distribution(target);
        let mut d = fixed_distribution(draft);
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        for n in [1usize, 5, 16, 32] {
            let out = run_vanilla_sd_with_mock(&mut t, &mut d, &[0u32], n, &cfg, &mut rng).unwrap();
            assert_eq!(
                out.len(),
                n,
                "expected exactly {n} tokens, got {}",
                out.len()
            );
        }
    }
}
