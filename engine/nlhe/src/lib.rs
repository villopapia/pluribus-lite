//! M2 tools for 6-max no-limit hold'em: abstraction config, betting rules with an exact
//! betting-tree counter, the blueprint memory estimator, a 7-card hand evaluator and river
//! equity. Nothing here allocates strategy tables or trains.

pub mod abstraction;
pub mod betting;
pub mod blueprint;
pub mod buckets;
pub mod config;
pub mod equity;
pub mod estimate;
pub mod eval;
pub mod eval_lut;
pub mod index;
