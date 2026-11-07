//! Regret / average-strategy storage and tabular policies.

use crate::fxhash::FxHashMap;
use crate::game::MAX_ACTIONS;

#[derive(Clone, Debug, PartialEq)]
pub struct Infoset {
    pub regret: Vec<f64>,
    /// Cumulative (possibly discounted) average-strategy weights (phi in the paper).
    pub strat_sum: Vec<f64>,
}

impl Infoset {
    pub fn new(n: usize) -> Self {
        Infoset { regret: vec![0.0; n], strat_sum: vec![0.0; n] }
    }

    #[inline]
    pub fn current_strategy(&self, out: &mut [f64]) {
        regret_matching(&self.regret, out);
    }
}

/// Regret matching: positive parts normalised, uniform if none positive.
#[inline]
pub fn regret_matching(regret: &[f64], out: &mut [f64]) {
    let n = regret.len();
    let mut pos_sum = 0.0;
    for &r in regret {
        if r > 0.0 {
            pos_sum += r;
        }
    }
    if pos_sum > 0.0 {
        for i in 0..n {
            out[i] = if regret[i] > 0.0 { regret[i] / pos_sum } else { 0.0 };
        }
    } else {
        let u = 1.0 / n as f64;
        for o in out.iter_mut().take(n) {
            *o = u;
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Table {
    pub map: FxHashMap<u64, Infoset>,
}

impl Table {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn get_or_insert(&mut self, key: u64, n: usize) -> &mut Infoset {
        self.map.entry(key).or_insert_with(|| Infoset::new(n))
    }

    /// Current regret-matched strategy at `key` (uniform if unseen), written to `out[..n]`.
    #[inline]
    pub fn strategy(&mut self, key: u64, n: usize, out: &mut [f64; MAX_ACTIONS]) {
        self.get_or_insert(key, n).current_strategy(&mut out[..n]);
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Normalised average strategy. Infosets with zero weight fall back to the
    /// current regret-matched strategy (as Pluribus does for post-flop rounds).
    pub fn average_policy(&self) -> TabularPolicy {
        let mut p = TabularPolicy::default();
        for (&k, inf) in &self.map {
            let s: f64 = inf.strat_sum.iter().sum();
            let v = if s > 0.0 {
                inf.strat_sum.iter().map(|x| x / s).collect()
            } else {
                let mut out = vec![0.0; inf.regret.len()];
                inf.current_strategy(&mut out);
                out
            };
            p.map.insert(k, v);
        }
        p
    }

    pub fn current_policy(&self) -> TabularPolicy {
        let mut p = TabularPolicy::default();
        for (&k, inf) in &self.map {
            let mut out = vec![0.0; inf.regret.len()];
            inf.current_strategy(&mut out);
            p.map.insert(k, out);
        }
        p
    }
}

/// Anything that maps an infoset to action probabilities.
pub trait Policy {
    fn action_probs(&self, key: u64, n: usize, out: &mut [f64]);
}

/// Explicit table of action probabilities; unseen infosets play uniformly.
#[derive(Clone, Debug, Default)]
pub struct TabularPolicy {
    pub map: FxHashMap<u64, Vec<f64>>,
}

impl Policy for TabularPolicy {
    fn action_probs(&self, key: u64, n: usize, out: &mut [f64]) {
        match self.map.get(&key) {
            Some(v) => {
                debug_assert_eq!(v.len(), n);
                out[..n].copy_from_slice(v);
            }
            None => {
                for o in out.iter_mut().take(n) {
                    *o = 1.0 / n as f64;
                }
            }
        }
    }
}

/// Uniform random policy.
pub struct UniformPolicy;

impl Policy for UniformPolicy {
    fn action_probs(&self, _key: u64, n: usize, out: &mut [f64]) {
        for o in out.iter_mut().take(n) {
            *o = 1.0 / n as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regret_matching_positive() {
        let mut out = [0.0; 3];
        regret_matching(&[1.0, -2.0, 3.0], &mut out);
        assert_eq!(out, [0.25, 0.0, 0.75]);
    }

    #[test]
    fn regret_matching_all_nonpositive_is_uniform() {
        let mut out = [0.0; 4];
        regret_matching(&[0.0, -1.0, -5.0, 0.0], &mut out);
        assert_eq!(out, [0.25; 4]);
    }

    #[test]
    fn average_policy_normalises_and_falls_back() {
        let mut t = Table::new();
        t.get_or_insert(1, 2).strat_sum = vec![1.0, 3.0];
        t.get_or_insert(2, 2).regret = vec![2.0, -1.0];
        let p = t.average_policy();
        let mut out = [0.0; 2];
        p.action_probs(1, 2, &mut out);
        assert_eq!(out, [0.25, 0.75]);
        p.action_probs(2, 2, &mut out);
        assert_eq!(out, [1.0, 0.0]);
        p.action_probs(99, 2, &mut out);
        assert_eq!(out, [0.5, 0.5]);
    }
}
