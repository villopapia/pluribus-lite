//! M2 design tools for 6-max no-limit hold'em: abstraction config, betting rules with an
//! exact betting-tree counter, and the blueprint memory estimator. Design only: nothing
//! here allocates strategy tables or trains.

pub mod betting;
pub mod config;
pub mod estimate;
