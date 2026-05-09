//! Token-level sampling helpers.
//!
//! These operate on plain `&[f32]` (or `&mut Vec<f32>`) so they can be unit-tested
//! without spinning up a model. The hot path in production runs against
//! candle tensors via `engine.rs`; that path delegates to these primitives after
//! materializing the relevant logits to host memory.

use crate::{Error, Result};
use rand::Rng;

/// Apply temperature, then softmax, in a numerically stable way.
///
/// `temperature == 0.0` is interpreted as greedy: returns a one-hot distribution
/// at the argmax (ties broken toward the lowest index).
pub fn softmax_with_temperature(logits: &[f32], temperature: f32) -> Result<Vec<f32>> {
    if logits.is_empty() {
        return Err(Error::Sampling("empty logits".into()));
    }
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(Error::Sampling(format!(
            "invalid temperature: {temperature}"
        )));
    }

    if temperature == 0.0 {
        let argmax = logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            })
            .0;
        let mut out = vec![0.0f32; logits.len()];
        out[argmax] = 1.0;
        return Ok(out);
    }

    let inv_t = 1.0 / temperature;
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = logits.iter().map(|&l| ((l - max) * inv_t).exp()).collect();
    let sum: f32 = out.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return Err(Error::Sampling(format!(
            "softmax produced non-positive sum: {sum}"
        )));
    }
    for v in out.iter_mut() {
        *v /= sum;
    }
    Ok(out)
}

/// Zero out the tail of the distribution so that the cumulative mass stays under
/// `top_p`, then renormalize. `top_p == 1.0` is a no-op.
pub fn top_p_filter(probs: &mut [f32], top_p: f32) -> Result<()> {
    if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
        return Err(Error::Sampling(format!("invalid top_p: {top_p}")));
    }
    if top_p >= 1.0 {
        return Ok(());
    }

    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cum = 0.0f32;
    let mut keep = vec![false; probs.len()];
    for (i, p) in &indexed {
        keep[*i] = true;
        cum += p;
        if cum >= top_p {
            break;
        }
    }
    let mut sum = 0.0f32;
    for (i, p) in probs.iter_mut().enumerate() {
        if !keep[i] {
            *p = 0.0;
        }
        sum += *p;
    }
    if sum <= 0.0 {
        return Err(Error::Sampling("top_p collapsed to zero mass".into()));
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }
    Ok(())
}

/// Sample a token index from a categorical distribution.
///
/// Assumes `probs` sums to 1 (small numerical drift is tolerated). Returns
/// `Err` if the distribution contains NaN or sums to <= 0.
pub fn sample_from_distribution<R: Rng + ?Sized>(rng: &mut R, probs: &[f32]) -> Result<usize> {
    if probs.is_empty() {
        return Err(Error::Sampling("empty distribution".into()));
    }
    let sum: f32 = probs.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(Error::Sampling(format!("invalid distribution sum: {sum}")));
    }
    let u: f32 = rng.gen::<f32>() * sum;
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        if !p.is_finite() || p < 0.0 {
            return Err(Error::Sampling(format!("invalid probability at {i}: {p}")));
        }
        acc += p;
        if u < acc {
            return Ok(i);
        }
    }
    // Numerical drift fallback: return the last non-zero index.
    Ok(probs.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::SeedableRng;

    #[test]
    fn softmax_sums_to_one() {
        let logits = [1.0, 2.0, 3.0];
        let p = softmax_with_temperature(&logits, 1.0).unwrap();
        let sum: f32 = p.iter().sum();
        assert_relative_eq!(sum, 1.0, max_relative = 1e-5);
    }

    #[test]
    fn greedy_temperature_returns_onehot() {
        let logits = [0.1, 5.0, 0.1, 4.99];
        let p = softmax_with_temperature(&logits, 0.0).unwrap();
        assert_eq!(p, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn top_p_zeros_tail() {
        let mut p = vec![0.5, 0.3, 0.15, 0.05];
        top_p_filter(&mut p, 0.8).unwrap();
        assert!(p[0] > 0.0 && p[1] > 0.0);
        assert_eq!(p[2], 0.0);
        assert_eq!(p[3], 0.0);
        let sum: f32 = p.iter().sum();
        assert_relative_eq!(sum, 1.0, max_relative = 1e-5);
    }

    #[test]
    fn sample_distribution_respects_mass() {
        // 100 % mass on index 2 → must always sample 2.
        let probs = vec![0.0, 0.0, 1.0, 0.0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for _ in 0..1000 {
            assert_eq!(sample_from_distribution(&mut rng, &probs).unwrap(), 2);
        }
    }

    #[test]
    fn sample_distribution_empirical_matches_expected() {
        let probs = vec![0.1, 0.2, 0.3, 0.4];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let n = 200_000;
        let mut counts = [0u32; 4];
        for _ in 0..n {
            counts[sample_from_distribution(&mut rng, &probs).unwrap()] += 1;
        }
        for (i, &expected) in probs.iter().enumerate() {
            let observed = counts[i] as f32 / n as f32;
            assert_relative_eq!(observed, expected, max_relative = 0.05);
        }
    }
}
