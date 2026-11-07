//! CFR / MCCFR correctness harness (milestone M1).
//!
//! * [`game`]: the `Game` trait every game implements.
//! * [`kuhn`], [`leduc`]: test games.
//! * [`cfr`]: vanilla CFR (full traversal, alternating updates).
//! * [`mccfr`]: external-sampling MCCFR with linear discounting and negative-regret
//!   pruning, following Algorithm 1 of the Pluribus supplement.
//! * [`br`]: expected values, best response and exploitability.
//! * [`checkpoint`]: binary checkpoint format used by both solvers.

pub mod br;
pub mod cfr;
pub mod checkpoint;
pub mod fxhash;
pub mod game;
pub mod intregret;
pub mod kuhn;
pub mod leduc;
pub mod mccfr;
pub mod rng;
pub mod table;
