//! Integer regret / average-strategy arithmetic for the blueprint (M3).
//!
//! The M1 solvers (`cfr`, `mccfr`) keep f64 regrets as the reference implementation. This
//! module holds the i32/u32 operations M3 will use. All rounding is **round half to even**,
//! computed exactly (integer arithmetic for discounting), which is:
//! * unbiased: rounding errors have mean 0, so repeated discounting does not drift;
//! * sign-preserving and odd-symmetric: discount(-r) = -discount(r), never flips a sign;
//! * platform independent: no float rounding modes, identical on Windows and Linux.
//!
//! Regret floor and prune threshold are the paper's values (unverified until the supplement
//! PDF check); our 6-max game uses the paper's chip units, so they apply unscaled.

pub const REGRET_FLOOR: i32 = -310_000_000;
pub const PRUNE_THRESHOLD: i32 = -300_000_000;

/// x * num / den rounded half to even, exactly. `den > 0`.
#[inline]
pub fn mul_div_round_even(x: i64, num: i64, den: i64) -> i64 {
    debug_assert!(den > 0 && num >= 0);
    let p = x as i128 * num as i128;
    let d = den as i128;
    let q = p.div_euclid(d); // floor
    let r = p.rem_euclid(d); // 0 <= r < d
    let twice = 2 * r;
    let q = if twice > d || (twice == d && q % 2 != 0) { q + 1 } else { q };
    q as i64
}

/// Linear-CFR discount step k (d = k / (k + 1)) applied to a regret.
#[inline]
pub fn discount_regret(r: i32, k: u64) -> i32 {
    mul_div_round_even(r as i64, k as i64, k as i64 + 1) as i32
}

/// Same for an average-strategy count.
#[inline]
pub fn discount_count(c: u32, k: u64) -> u32 {
    mul_div_round_even(c as i64, k as i64, k as i64 + 1) as u32
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RegretStats {
    /// Updates clamped at i32::MAX (must stay 0 in a healthy run).
    pub saturated: u64,
    /// Updates clamped at the floor.
    pub floored: u64,
}

/// r += round_half_even(delta), clamped to [floor, i32::MAX].
#[inline]
pub fn add_regret(r: &mut i32, delta: f64, floor: i32, stats: &mut RegretStats) {
    let v = *r as i64 + delta.round_ties_even() as i64;
    *r = if v > i32::MAX as i64 {
        stats.saturated += 1;
        i32::MAX
    } else if v < floor as i64 {
        stats.floored += 1;
        floor
    } else {
        v as i32
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn half_to_even_cases() {
        assert_eq!(mul_div_round_even(5, 1, 2), 2); // 2.5 -> 2
        assert_eq!(mul_div_round_even(7, 1, 2), 4); // 3.5 -> 4
        assert_eq!(mul_div_round_even(-5, 1, 2), -2);
        assert_eq!(mul_div_round_even(-7, 1, 2), -4);
        assert_eq!(mul_div_round_even(10, 2, 3), 7); // 6.67 -> 7
        assert_eq!(mul_div_round_even(-10, 2, 3), -7);
        assert_eq!(mul_div_round_even(1, 1, 3), 0);
        assert_eq!(mul_div_round_even(i32::MIN as i64, 39, 40), -2_093_796_557); // -2093796556.8
    }

    #[test]
    fn symmetric_sign_preserving_contracting() {
        let mut rng = Rng::new(1);
        for _ in 0..100_000 {
            let r = (rng.next_u64() as i64 % 2_000_000_000) as i32;
            let k = 1 + rng.next_u64() % 1000;
            let d = discount_regret(r, k);
            assert_eq!(discount_regret(-r, k), -d);
            assert!(d == 0 || d.signum() == r.signum());
            assert!(d.unsigned_abs() <= r.unsigned_abs());
        }
    }

    /// Pluribus-style schedule: discount steps k = 1..K. The exact product telescopes to
    /// 1 / (K + 1). Integer half-even rounding must stay unbiased (mean error ~ 0) and within
    /// the worst-case bound sum_j 0.5 * (j + 1) / (K + 1); floor rounding drifts.
    #[test]
    fn repeated_discounting_does_not_drift() {
        const K: u64 = 40;
        let mut rng = Rng::new(2);
        let n = 200_000;
        let bound: f64 = (1..=K).map(|j| 0.5 * (j + 1) as f64 / (K + 1) as f64).sum();
        let (mut err_even, mut err_floor, mut max_abs) = (0.0f64, 0.0f64, 0.0f64);
        for _ in 0..n {
            let r0 = (rng.next_u64() % 600_000_001) as i64 - 300_000_000;
            let mut even = r0 as i32;
            let mut floor = r0;
            for k in 1..=K {
                even = discount_regret(even, k);
                floor = (floor * k as i64).div_euclid(k as i64 + 1);
            }
            let exact = r0 as f64 / (K + 1) as f64;
            let e = even as f64 - exact;
            err_even += e;
            err_floor += floor as f64 - exact;
            max_abs = max_abs.max(e.abs());
        }
        let (mean_even, mean_floor) = (err_even / n as f64, err_floor / n as f64);
        assert!(mean_even.abs() < 0.02, "half-even mean error {mean_even}");
        assert!(max_abs <= bound, "max error {max_abs} > bound {bound}");
        // the test is sensitive: floor division visibly drifts negative
        assert!(mean_floor < -1.0, "floor mean error {mean_floor}");
    }

    /// Interleaved updates + discounts against an f64 shadow: absolute error stays bounded.
    #[test]
    fn interleaved_updates_track_f64_shadow() {
        let mut rng = Rng::new(3);
        let mut stats = RegretStats::default();
        let mut worst = 0.0f64;
        for _ in 0..2_000 {
            let mut r: i32 = 0;
            let mut shadow = 0.0f64;
            for t in 1..=400u64 {
                let delta = (rng.next_f64() - 0.45) * 20_000.0;
                add_regret(&mut r, delta, i32::MIN, &mut stats);
                shadow += delta;
                if t % 10 == 0 {
                    let k = t / 10;
                    r = discount_regret(r, k);
                    shadow *= k as f64 / (k + 1) as f64;
                }
            }
            worst = worst.max((r as f64 - shadow).abs());
        }
        // each update rounds by <= 0.5 and discounting only shrinks old errors
        assert!(worst < 50.0, "worst abs error {worst}");
        assert_eq!(stats.saturated, 0);
    }

    #[test]
    fn floor_and_saturation() {
        let mut s = RegretStats::default();
        let mut r = REGRET_FLOOR + 10;
        add_regret(&mut r, -100.0, REGRET_FLOOR, &mut s);
        assert_eq!((r, s.floored), (REGRET_FLOOR, 1));
        let mut r = i32::MAX - 5;
        add_regret(&mut r, 100.0, REGRET_FLOOR, &mut s);
        assert_eq!((r, s.saturated), (i32::MAX, 1));
        let mut r = 0;
        add_regret(&mut r, 2.5, REGRET_FLOOR, &mut s);
        assert_eq!(r, 2); // half to even
        add_regret(&mut r, 0.5, REGRET_FLOOR, &mut s);
        assert_eq!(r, 2);
        assert!(PRUNE_THRESHOLD > REGRET_FLOOR);
    }

    #[test]
    fn counts_discount() {
        assert_eq!(discount_count(5, 1), 2);
        assert_eq!(discount_count(u32::MAX, 1), 2_147_483_648); // 2147483647.5 -> even
    }
}
