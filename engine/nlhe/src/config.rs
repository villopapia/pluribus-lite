//! Abstraction configuration: bucket counts and bet-size sets, loaded from a small
//! line-based text format (no dependencies):
//!
//! ```text
//! # comment
//! name = small
//! buckets = 169, 200, 200, 200          # preflop, flop, turn, river
//! allin = true                          # all-in always available
//! preflop.0 = 0.4, 0.5, 0.75            # sizes at raise depth 0 (open), as pot fractions
//! preflop.1 = 1.0                       # depth 1 (3-bet); deeper depths -> all-in only
//! flop.0 = 0.5, 1.0
//! multiway.flop.0 = 0.5                 # override when >= 3 players are still in the hand
//! avg_strategy_streets = preflop        # streets that store an average strategy
//! ```
//!
//! A size f means "raise to max_bet + f * (pot + to_call)": f = 1 is a pot-sized bet or raise.
//! This matches `bet_frac_pot` in analysis/m0_parse.py. Raise depth = number of voluntary
//! bets/raises already made on the street (preflop blinds do not count). Depths beyond the
//! listed ones allow only all-in (if `allin = true`).

use std::fmt;

pub const STREETS: [&str; 4] = ["preflop", "flop", "turn", "river"];

#[derive(Clone, Debug, PartialEq)]
pub struct AbstractionConfig {
    pub name: String,
    pub buckets: [u32; 4],
    pub allin: bool,
    /// sizes[street][depth] used heads-up (2 players in the hand).
    pub sizes: [Vec<Vec<f64>>; 4],
    /// Post-flop overrides per depth when >= 3 players are in the hand. A depth that is not
    /// set (`None`) falls back to `sizes`; an explicitly empty list means all-in only.
    /// Not allowed preflop, where every open is "multiway" by player count.
    pub multiway: [Vec<Option<Vec<f64>>>; 4],
    /// `<street>.<d>+ = sizes`: these sizes at depth d and every deeper depth (unless a
    /// specific depth is listed). Used for the paper's turn/river rule "later raises {1x}".
    pub repeat_from: [Option<(usize, Vec<f64>)>; 4],
    pub avg_strategy_streets: [bool; 4],
    /// Who may flat-call the big blind in an unraised preflop pot (BB checking is always allowed).
    pub limp: Limp,
    /// Post-flop, with at least this many players in the hand, no sized bets (check / call /
    /// fold / all-in only). 0 = off.
    pub crowded_players: u8,
    /// How river buckets are obtained at training time.
    pub river_method: RiverMethod,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RiverMethod {
    /// Equity computed per iteration (lazy, only hands that reach the river), then the
    /// bucket from equity cut-offs. No table memory.
    OnTheFly,
    /// u8/u16 bucket per (hole, river board): 2,809,475,760 entries (2.62 GiB at u8).
    /// Meant for machines where that memory is free (e.g. HPC nodes).
    PrecomputedTable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limp {
    None,
    /// Only the small blind may complete.
    Sb,
    All,
}

#[derive(Debug, PartialEq)]
pub struct ConfigError(pub String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "config error: {}", self.0)
    }
}

fn street_index(s: &str) -> Result<usize, ConfigError> {
    STREETS
        .iter()
        .position(|&x| x == s)
        .ok_or_else(|| ConfigError(format!("unknown street {s:?}")))
}

fn parse_list(v: &str) -> Result<Vec<f64>, ConfigError> {
    if v.trim().is_empty() {
        return Ok(vec![]);
    }
    v.split(',')
        .map(|x| {
            let f: f64 = x.trim().parse().map_err(|_| ConfigError(format!("bad number {x:?}")))?;
            if !(f > 0.0 && f.is_finite()) {
                return Err(ConfigError(format!("size must be > 0, got {f}")));
            }
            Ok(f)
        })
        .collect()
}

fn set_depth(v: &mut Vec<Vec<f64>>, depth: usize, sizes: Vec<f64>) {
    if v.len() <= depth {
        v.resize(depth + 1, vec![]);
    }
    v[depth] = sizes;
}

impl AbstractionConfig {
    pub fn parse(text: &str) -> Result<Self, ConfigError> {
        let mut c = AbstractionConfig {
            name: "unnamed".into(),
            buckets: [169, 200, 200, 200],
            allin: true,
            sizes: Default::default(),
            multiway: Default::default(),
            repeat_from: Default::default(),
            avg_strategy_streets: [true, false, false, false],
            limp: Limp::All,
            crowded_players: 0,
            river_method: RiverMethod::OnTheFly,
        };
        for (ln, raw) in text.lines().enumerate() {
            let line = raw.split('#').next().unwrap().trim();
            if line.is_empty() {
                continue;
            }
            let (k, v) = line
                .split_once('=')
                .ok_or_else(|| ConfigError(format!("line {}: expected key = value", ln + 1)))?;
            let (k, v) = (k.trim(), v.trim());
            match k {
                "name" => c.name = v.to_string(),
                "limp" => {
                    c.limp = match v {
                        "none" => Limp::None,
                        "sb" => Limp::Sb,
                        "all" => Limp::All,
                        _ => return Err(ConfigError(format!("limp: expected none|sb|all, got {v:?}"))),
                    }
                }
                "river_method" => {
                    c.river_method = match v {
                        "on_the_fly" => RiverMethod::OnTheFly,
                        "precomputed_table" => RiverMethod::PrecomputedTable,
                        _ => {
                            return Err(ConfigError(format!(
                                "river_method: expected on_the_fly|precomputed_table, got {v:?}"
                            )))
                        }
                    }
                }
                "crowded_players" => {
                    c.crowded_players = v.parse().map_err(|_| ConfigError(format!("crowded_players: {v:?}")))?
                }
                "allin" => {
                    c.allin = v.parse().map_err(|_| ConfigError(format!("allin: bad bool {v:?}")))?
                }
                "buckets" => {
                    let b: Vec<u32> = v
                        .split(',')
                        .map(|x| x.trim().parse::<u32>())
                        .collect::<Result<_, _>>()
                        .map_err(|_| ConfigError(format!("buckets: {v:?}")))?;
                    if b.len() != 4 || b.contains(&0) {
                        return Err(ConfigError("buckets needs 4 positive values".into()));
                    }
                    c.buckets = [b[0], b[1], b[2], b[3]];
                }
                "avg_strategy_streets" => {
                    c.avg_strategy_streets = [false; 4];
                    for s in v.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                        c.avg_strategy_streets[street_index(s)?] = true;
                    }
                }
                _ => {
                    if let Some(base) = k.strip_suffix('+') {
                        let parts: Vec<&str> = base.split('.').collect();
                        let (street, depth) = match parts.as_slice() {
                            [s, d] => (*s, *d),
                            _ => return Err(ConfigError(format!("bad repeat key {k:?}"))),
                        };
                        let si = street_index(street)?;
                        let d: usize = depth.parse().map_err(|_| ConfigError(format!("bad depth in {k:?}")))?;
                        c.repeat_from[si] = Some((d, parse_list(v)?));
                        continue;
                    }
                    let parts: Vec<&str> = k.split('.').collect();
                    let (mw, street, depth) = match parts.as_slice() {
                        [s, d] => (false, *s, *d),
                        ["multiway", s, d] => (true, *s, *d),
                        _ => return Err(ConfigError(format!("unknown key {k:?}"))),
                    };
                    let si = street_index(street)?;
                    let d: usize = depth.parse().map_err(|_| ConfigError(format!("bad depth in {k:?}")))?;
                    let sizes = parse_list(v)?;
                    if mw {
                        if si == 0 {
                            return Err(ConfigError("multiway overrides are post-flop only".into()));
                        }
                        let m = &mut c.multiway[si];
                        if m.len() <= d {
                            m.resize(d + 1, None);
                        }
                        m[d] = Some(sizes);
                    } else {
                        set_depth(&mut c.sizes[si], d, sizes);
                    }
                }
            }
        }
        Ok(c)
    }

    /// Bet sizes (pot fractions) at this street / raise depth with `n_in_hand` players left.
    /// Empty slice = no sized raises (all-in only, if enabled).
    pub fn sizes_for(&self, street: usize, depth: usize, n_in_hand: usize) -> &[f64] {
        if street > 0 && self.crowded_players > 0 && n_in_hand >= self.crowded_players as usize {
            return &[];
        }
        if street > 0 && n_in_hand >= 3 {
            if let Some(Some(v)) = self.multiway[street].get(depth) {
                return v;
            }
        }
        if let Some(v) = self.sizes[street].get(depth) {
            if !v.is_empty() || self.repeat_from[street].as_ref().is_none_or(|(d, _)| depth < *d) {
                return v;
            }
        }
        match &self.repeat_from[street] {
            Some((d, v)) if depth >= *d => v,
            _ => &[],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TXT: &str = "
        name = t
        buckets = 169, 10, 20, 30
        allin = false
        preflop.0 = 0.5, 1.0   # opens
        preflop.1 = 2
        flop.0 = 0.25,0.5
        multiway.flop.0 = 0.5
        avg_strategy_streets = preflop, river
    ";

    #[test]
    fn parses_all_fields() {
        let c = AbstractionConfig::parse(TXT).unwrap();
        assert_eq!(c.name, "t");
        assert_eq!(c.buckets, [169, 10, 20, 30]);
        assert!(!c.allin);
        assert_eq!(c.sizes_for(0, 0, 6), &[0.5, 1.0]);
        assert_eq!(c.sizes_for(0, 1, 2), &[2.0]);
        assert!(c.sizes_for(0, 2, 2).is_empty());
        assert_eq!(c.sizes_for(1, 0, 2), &[0.25, 0.5]);
        assert_eq!(c.sizes_for(1, 0, 3), &[0.5]);
        assert!(c.sizes_for(1, 1, 3).is_empty());
        assert!(c.sizes_for(3, 0, 2).is_empty());
        assert_eq!(c.avg_strategy_streets, [true, false, false, true]);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(AbstractionConfig::parse("foo = 1").is_err());
        assert!(AbstractionConfig::parse("flop.0 = 0.5, -1").is_err());
        assert!(AbstractionConfig::parse("buckets = 1, 2, 3").is_err());
        assert!(AbstractionConfig::parse("bogus.0 = 1").is_err());
        assert!(AbstractionConfig::parse("no equals sign").is_err());
        assert!(AbstractionConfig::parse("multiway.preflop.1 = 1.0").is_err());
        assert!(AbstractionConfig::parse("limp = maybe").is_err());
        assert!(AbstractionConfig::parse("river_method = cached").is_err());
        assert_eq!(AbstractionConfig::parse("").unwrap().river_method, RiverMethod::OnTheFly);
        assert_eq!(
            AbstractionConfig::parse("river_method = precomputed_table").unwrap().river_method,
            RiverMethod::PrecomputedTable
        );
    }

    #[test]
    fn repeat_from_depth() {
        let c = AbstractionConfig::parse("turn.0 = 0.5, 1.0
turn.1+ = 1.0").unwrap();
        assert_eq!(c.sizes_for(2, 0, 2), &[0.5, 1.0]);
        assert_eq!(c.sizes_for(2, 1, 2), &[1.0]);
        assert_eq!(c.sizes_for(2, 5, 6), &[1.0]);
        assert!(c.sizes_for(3, 1, 2).is_empty());
        assert!(AbstractionConfig::parse("turn+ = 1.0").is_err());
    }

    #[test]
    fn limp_and_crowded() {
        let c = AbstractionConfig::parse("limp = sb
crowded_players = 4
flop.0 = 0.5").unwrap();
        assert_eq!(c.limp, Limp::Sb);
        assert_eq!(c.sizes_for(1, 0, 3), &[0.5]);
        assert!(c.sizes_for(1, 0, 4).is_empty());
        assert_eq!(AbstractionConfig::parse("").unwrap().limp, Limp::All);
    }

    #[test]
    fn multiway_falls_back_per_depth_and_empty_means_allin_only() {
        let c = AbstractionConfig::parse(
            "flop.0 = 0.5, 1.0
             flop.1 = 1.0
             flop.2 = 2.0
             multiway.flop.1 = 0.5
             multiway.flop.2 =",
        )
        .unwrap();
        assert_eq!(c.sizes_for(1, 0, 3), &[0.5, 1.0]); // depth 0 not overridden
        assert_eq!(c.sizes_for(1, 1, 3), &[0.5]);
        assert!(c.sizes_for(1, 2, 3).is_empty()); // explicit empty
        assert_eq!(c.sizes_for(1, 2, 2), &[2.0]); // heads-up unaffected
    }
}
