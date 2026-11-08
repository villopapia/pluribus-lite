//! Real-time search primitives shared by the toy games and NLHE (M4, `docs/m4_design.md`).
//!
//! * [`bayes`]: Bayes' rule on a belief range (stage 1, Supp p. 21).
//! * [`bias`]: the four continuation strategies at a leaf (stage 3, Supp p. 24).
//! * [`sample_fixed`]: compression of a continuation strategy to one sampled action per abstract
//!   infoset (Supp p. 24), computed on demand from a hash instead of a stored table (D-SRCH-6).

/// Action class used to bias continuation strategies. A check counts as a call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActKind {
    Fold,
    Call,
    Raise,
}

/// Continuation strategies at a leaf: 0 blueprint, 1 fold-, 2 call-, 3 raise-biased.
pub const N_CONT: usize = 4;
/// Factor applied to the biased action class before renormalising (Supp p. 24).
pub const BIAS: f64 = 5.0;

/// Bayes' rule: `range[h] <- range[h] * lik[h]`, renormalised. Returns the probability of the
/// observation under the prior (the range must sum to 1). If that probability is 0, the range is
/// left unchanged and 0 is returned so the caller can report it (D-SRCH-2).
pub fn bayes(range: &mut [f64], lik: &[f64]) -> f64 {
    assert_eq!(range.len(), lik.len());
    let z: f64 = range.iter().zip(lik).map(|(r, l)| r * l).sum();
    if z > 0.0 {
        for (r, l) in range.iter_mut().zip(lik) {
            *r = *r * l / z;
        }
    }
    z
}

/// Continuation strategy `k` from blueprint probabilities: the probabilities of the actions of
/// the biased class (k = 1 fold, 2 call, 3 every raise) are multiplied by [`BIAS`], then all are
/// renormalised; k = 0 is the blueprint. If no legal action has that class, the strategy is
/// unchanged (D-SRCH-5).
pub fn bias(probs: &[f64], kinds: &[ActKind], k: usize, out: &mut [f64]) {
    let target = match k {
        0 => None,
        1 => Some(ActKind::Fold),
        2 => Some(ActKind::Call),
        3 => Some(ActKind::Raise),
        _ => panic!("continuation strategy {k} out of range"),
    };
    let n = probs.len();
    let mut sum = 0.0;
    for i in 0..n {
        let w = if Some(kinds[i]) == target { BIAS } else { 1.0 };
        out[i] = probs[i] * w;
        sum += out[i];
    }
    if sum > 0.0 {
        for o in out.iter_mut().take(n) {
            *o /= sum;
        }
    }
}

#[inline]
fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One action sampled from `probs` for (seed, abstract infoset `key`, strategy `k`): the same
/// arguments always give the same action, as a stored table would. Zero-probability actions are
/// never returned.
pub fn sample_fixed(seed: u64, key: u64, k: usize, probs: &[f64]) -> usize {
    let h = splitmix(splitmix(splitmix(seed) ^ key) ^ k as u64);
    let u = (h >> 11) as f64 / (1u64 << 53) as f64;
    let total: f64 = probs.iter().sum();
    let x = u * total;
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if x < acc {
            return i;
        }
    }
    // only reachable through rounding at the top end
    probs.iter().rposition(|&p| p > 0.0).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ActKind::*;

    fn close(a: &[f64], b: &[f64]) {
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-12, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn bias_matches_hand_computed_values() {
        let p = [0.2, 0.5, 0.3];
        let kinds = [Fold, Call, Raise];
        let mut out = [0.0; 3];
        bias(&p, &kinds, 0, &mut out);
        close(&out, &p);
        bias(&p, &kinds, 1, &mut out);
        close(&out, &[1.0 / 1.8, 0.5 / 1.8, 0.3 / 1.8]);
        bias(&p, &kinds, 2, &mut out);
        close(&out, &[0.2 / 3.0, 2.5 / 3.0, 0.3 / 3.0]);
        // every raise size is biased
        let p4 = [0.2, 0.5, 0.1, 0.2];
        let mut out4 = [0.0; 4];
        bias(&p4, &[Fold, Call, Raise, Raise], 3, &mut out4);
        close(&out4, &[0.2 / 2.2, 0.5 / 2.2, 0.5 / 2.2, 1.0 / 2.2]);
    }

    #[test]
    fn bias_toward_an_illegal_action_leaves_the_strategy_unchanged() {
        // facing no bet: check or bet, no fold
        let p = [0.6, 0.4];
        let mut out = [0.0; 2];
        bias(&p, &[Call, Raise], 1, &mut out);
        close(&out, &p);
        // only fold and call legal (raise cap)
        bias(&[0.3, 0.7], &[Fold, Call], 3, &mut out);
        close(&out, &[0.3, 0.7]);
    }

    #[test]
    fn sample_fixed_is_deterministic_and_has_the_right_frequencies() {
        let p = [0.1, 0.6, 0.3];
        assert_eq!(sample_fixed(7, 42, 2, &p), sample_fixed(7, 42, 2, &p));
        let n = 200_000u64;
        let mut cnt = [0u64; 3];
        for key in 0..n {
            cnt[sample_fixed(7, key, 1, &p)] += 1;
        }
        for i in 0..3 {
            let f = cnt[i] as f64 / n as f64;
            assert!((f - p[i]).abs() < 0.005, "action {i}: {f}");
        }
        // a degenerate distribution always gives its action
        for key in 0..1000 {
            assert_eq!(sample_fixed(3, key, 0, &[0.0, 1.0, 0.0]), 1);
        }
        // strategies k are sampled independently: two fair coins agree about half the time
        let agree = (0..n).filter(|&key| sample_fixed(9, key, 0, &[0.5, 0.5]) == sample_fixed(9, key, 1, &[0.5, 0.5])).count();
        let f = agree as f64 / n as f64;
        assert!((f - 0.5).abs() < 0.005, "agreement {f}");
    }

    #[test]
    fn bayes_updates_and_keeps_the_prior_on_impossible_observations() {
        let mut r = [0.5, 0.5];
        let z = bayes(&mut r, &[1.0, 0.5]);
        assert!((z - 0.75).abs() < 1e-15);
        close(&r, &[2.0 / 3.0, 1.0 / 3.0]);
        let z = bayes(&mut r, &[0.0, 0.0]);
        assert_eq!(z, 0.0);
        close(&r, &[2.0 / 3.0, 1.0 / 3.0]);
    }
}
