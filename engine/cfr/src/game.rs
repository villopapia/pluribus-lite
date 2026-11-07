//! The `Game` trait: an extensive-form game with chance, perfect recall and
//! integer infoset keys. Actions are small integers indexing `legal_actions`.

use crate::rng::Rng;

pub type Action = u8;
pub const MAX_ACTIONS: usize = 8;
pub const MAX_CHANCE: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    Terminal,
    Chance,
    Player(usize),
}

/// Fixed-capacity action list (no heap allocation in the hot loop).
#[derive(Clone, Copy, Debug)]
pub struct ActionList {
    len: usize,
    items: [Action; MAX_ACTIONS],
}

impl ActionList {
    pub fn new() -> Self {
        ActionList { len: 0, items: [0; MAX_ACTIONS] }
    }
    pub fn from_slice(s: &[Action]) -> Self {
        let mut l = Self::new();
        for &a in s {
            l.push(a);
        }
        l
    }
    #[inline]
    pub fn push(&mut self, a: Action) {
        self.items[self.len] = a;
        self.len += 1;
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline]
    pub fn as_slice(&self) -> &[Action] {
        &self.items[..self.len]
    }
}

impl Default for ActionList {
    fn default() -> Self {
        Self::new()
    }
}

/// Fixed-capacity list of chance outcomes with probabilities.
#[derive(Clone, Copy, Debug)]
pub struct ChanceList {
    len: usize,
    actions: [Action; MAX_CHANCE],
    probs: [f64; MAX_CHANCE],
}

impl ChanceList {
    pub fn new() -> Self {
        ChanceList { len: 0, actions: [0; MAX_CHANCE], probs: [0.0; MAX_CHANCE] }
    }
    #[inline]
    pub fn push(&mut self, a: Action, p: f64) {
        self.actions[self.len] = a;
        self.probs[self.len] = p;
        self.len += 1;
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline]
    pub fn actions(&self) -> &[Action] {
        &self.actions[..self.len]
    }
    #[inline]
    pub fn probs(&self) -> &[f64] {
        &self.probs[..self.len]
    }
}

impl Default for ChanceList {
    fn default() -> Self {
        Self::new()
    }
}

pub trait Game {
    type State: Clone;

    fn name(&self) -> &'static str;
    fn num_players(&self) -> usize;
    /// Number of betting rounds (Kuhn 1, Leduc 2, hold'em 4).
    fn num_rounds(&self) -> u8;
    /// Largest |utility| any terminal can give; used to scale pruning thresholds.
    fn max_abs_utility(&self) -> f64;

    fn root(&self) -> Self::State;
    fn kind(&self, s: &Self::State) -> NodeKind;
    fn utility(&self, s: &Self::State, player: usize) -> f64;
    fn chance_outcomes(&self, s: &Self::State) -> ChanceList;
    fn legal_actions(&self, s: &Self::State) -> ActionList;
    fn apply(&self, s: &Self::State, a: Action) -> Self::State;
    /// Key of the acting player's information set. Must be unique across players.
    fn infoset_key(&self, s: &Self::State) -> u64;
    /// Current betting round (0-based).
    fn round(&self, s: &Self::State) -> u8;

    fn sample_chance(&self, s: &Self::State, rng: &mut Rng) -> Action {
        let c = self.chance_outcomes(s);
        c.actions()[rng.sample_index(c.probs())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_list_basics() {
        let mut l = ActionList::new();
        assert!(l.is_empty());
        l.push(2);
        l.push(5);
        assert_eq!(l.as_slice(), &[2, 5]);
        assert_eq!(ActionList::from_slice(&[1, 2, 3]).len(), 3);
    }

    #[test]
    fn chance_list_basics() {
        let mut c = ChanceList::new();
        c.push(0, 0.25);
        c.push(3, 0.75);
        assert_eq!(c.actions(), &[0, 3]);
        assert_eq!(c.probs(), &[0.25, 0.75]);
    }
}
