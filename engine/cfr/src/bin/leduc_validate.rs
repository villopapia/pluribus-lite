//! M4 validation on Leduc (`docs/m4_validation.md`):
//! (a) depth-limited search with the 4 continuation strategies vs the blueprint alone;
//! (b) blueprint-only leaves vs the 4 strategies.
//!
//! `cargo run --release --bin leduc_validate`
//!
//! Blueprints: M1's MCCFR (Pluribus schedule scaled) after 2k / 20k / 200k iterations, seed 1.
//! Agents (Algorithm 2, `cfr::nested`), each evaluated exactly: exploitability of the whole
//! strategy the agent plays (both seats, exact best response) and its head-to-head value against
//! the blueprint (average of both seats). Solver seeds 1-3; mean and range reported. Each agent
//! plays the final iteration (paper) and, as a diagnostic, the weighted average.
//! `--no-compress`: exact biased continuation strategies instead of one sampled action per
//! infoset (diagnostic; the paper compresses, D-SRCH-6).

use cfr::br::{best_response_value, expected_values, exploitability};
use cfr::cfr::Cfr;
use cfr::game::{Action, ActionList, ChanceList, Game, NodeKind};
use cfr::leduc::LeducState;
use cfr::table::Policy;
use cfr::leduc::Leduc;
use cfr::mccfr::{Mccfr, MccfrConfig};
use cfr::nested::{agent_policy, Agent, SearchConfig};
use cfr::subgame_solve::Solver;
use cfr::table::TabularPolicy;
use std::time::Instant;

/// Leduc whose round-2 infosets forget the own card's rank: only "paired with the board" or not
/// (a lossy blueprint abstraction, like Pluribus' 200 buckets, while search is lossless).
struct AbsLeduc;

impl Game for AbsLeduc {
    type State = LeducState;
    fn name(&self) -> &'static str {
        "leduc-abs"
    }
    fn num_players(&self) -> usize {
        2
    }
    fn num_rounds(&self) -> u8 {
        2
    }
    fn max_abs_utility(&self) -> f64 {
        13.0
    }
    fn root(&self) -> LeducState {
        Leduc.root()
    }
    fn kind(&self, s: &LeducState) -> NodeKind {
        Leduc.kind(s)
    }
    fn utility(&self, s: &LeducState, p: usize) -> f64 {
        Leduc.utility(s, p)
    }
    fn chance_outcomes(&self, s: &LeducState) -> ChanceList {
        Leduc.chance_outcomes(s)
    }
    fn legal_actions(&self, s: &LeducState) -> ActionList {
        Leduc.legal_actions(s)
    }
    fn apply(&self, s: &LeducState, a: Action) -> LeducState {
        Leduc.apply(s, a)
    }
    fn infoset_key(&self, s: &LeducState) -> u64 {
        abs_key(s)
    }
    fn round(&self, s: &LeducState) -> u8 {
        s.round
    }
}

fn abs_key(s: &LeducState) -> u64 {
    let k = Leduc.infoset_key(s);
    if s.round == 0 {
        return k;
    }
    let paired = s.cards[s.to_act as usize] / 2 == s.board / 2;
    (k & !3) | paired as u64 // low 2 bits held the own rank (key_for)
}

/// The abstract policy on Leduc's own infosets.
fn expand(abs: &TabularPolicy) -> TabularPolicy {
    fn walk(s: &LeducState, abs: &TabularPolicy, out: &mut TabularPolicy) {
        let g = Leduc;
        match g.kind(s) {
            NodeKind::Terminal => {}
            NodeKind::Chance => g.chance_outcomes(s).actions().iter().for_each(|&a| walk(&g.apply(s, a), abs, out)),
            NodeKind::Player(_) => {
                let n = g.legal_actions(s).len();
                let mut p = vec![0.0; n];
                abs.action_probs(abs_key(s), n, &mut p);
                out.map.insert(g.infoset_key(s), p);
                g.legal_actions(s).as_slice().iter().for_each(|&a| walk(&g.apply(s, a), abs, out));
            }
        }
    }
    let mut out = TabularPolicy::default();
    walk(&Leduc.root(), abs, &mut out);
    out
}

/// Leduc where player `fix`'s round-1 decisions are chance nodes over `pol`: a best responder
/// limited to deviating on round 2 (diagnostic for unsafe search).
struct FixRound1<'a> {
    pol: &'a TabularPolicy,
    fix: usize,
}

impl Game for FixRound1<'_> {
    type State = LeducState;
    fn name(&self) -> &'static str {
        "leduc-fix-r1"
    }
    fn num_players(&self) -> usize {
        2
    }
    fn num_rounds(&self) -> u8 {
        2
    }
    fn max_abs_utility(&self) -> f64 {
        13.0
    }
    fn root(&self) -> LeducState {
        Leduc.root()
    }
    fn kind(&self, s: &LeducState) -> NodeKind {
        match Leduc.kind(s) {
            NodeKind::Player(p) if p == self.fix && s.round == 0 => NodeKind::Chance,
            k => k,
        }
    }
    fn utility(&self, s: &LeducState, p: usize) -> f64 {
        Leduc.utility(s, p)
    }
    fn chance_outcomes(&self, s: &LeducState) -> ChanceList {
        if let NodeKind::Player(_) = Leduc.kind(s) {
            let acts = Leduc.legal_actions(s);
            let mut p = [0.0; 8];
            self.pol.action_probs(Leduc.infoset_key(s), acts.len(), &mut p);
            let mut c = ChanceList::new();
            for (i, &a) in acts.as_slice().iter().enumerate() {
                if p[i] > 0.0 {
                    c.push(a, p[i]);
                }
            }
            return c;
        }
        Leduc.chance_outcomes(s)
    }
    fn legal_actions(&self, s: &LeducState) -> ActionList {
        Leduc.legal_actions(s)
    }
    fn apply(&self, s: &LeducState, a: Action) -> LeducState {
        Leduc.apply(s, a)
    }
    fn infoset_key(&self, s: &LeducState) -> u64 {
        Leduc.infoset_key(s)
    }
    fn round(&self, s: &LeducState) -> u8 {
        s.round
    }
}

/// Exploitability when the best responder plays the blueprint on round 1.
fn round2_exploitability(pol: &TabularPolicy, bp: &TabularPolicy) -> f64 {
    (0..2).map(|b| best_response_value(&FixRound1 { pol: bp, fix: b }, pol, b)).sum::<f64>() / 2.0
}

fn head_to_head(agent: &TabularPolicy, owner: &std::collections::HashMap<u64, usize>, bp: &TabularPolicy) -> f64 {
    let mut total = 0.0;
    for seat in 0..2 {
        let mut mix = bp.clone();
        for (k, v) in &agent.map {
            if owner[k] == seat {
                mix.map.insert(*k, v.clone());
            }
        }
        total += expected_values(&Leduc, &mix)[seat];
    }
    total / 2.0
}

fn main() {
    let t0 = Instant::now();
    let variants: [(&str, usize, bool, bool); 6] = [
        ("A  search r1+r2, 4 continuations", 4, true, true),
        ("B  search r1+r2, blueprint leaves", 1, true, true),
        ("C  search r1 only, 4 continuations", 4, true, false),
        ("D  search r1 only, blueprint leaves", 1, true, false),
        ("E  literal Alg. 2 (search r2 only)", 4, false, true),
        ("F  search r1 only, 4 continuations, r2 follows the chosen continuation (diagnostic)", 4, true, false),
    ];
    let solver_sets: [(&str, fn(u64) -> (Solver, Solver)); 2] = [
        ("paper rule: MCCFR 50k early, vector 1k late", |s| (Solver::Mccfr { iters: 50_000, seed: s }, Solver::Vector { iters: 1000, seed: s })),
        ("vector 1k everywhere", |s| (Solver::Vector { iters: 1000, seed: s }, Solver::Vector { iters: 1000, seed: s })),
    ];
    println!("Leduc rounds are r1 (before the board) and r2. Values in chips per hand (ante 1).");
    let mut blueprints: Vec<(String, TabularPolicy)> = [2_000u64, 20_000, 200_000]
        .into_iter()
        .map(|n| {
            let mut m = Mccfr::new(Leduc, MccfrConfig::pluribus_scaled(n, 13.0, 1));
            m.run(n);
            (format!("MCCFR {n} iterations (no abstraction)"), m.average_policy())
        })
        .collect();
    let mut abs = Cfr::new(AbsLeduc);
    abs.run(5_000);
    blueprints.push(("abstracted (round 2 sees only paired / unpaired), CFR 5000 iterations".into(), expand(&abs.average_policy())));
    for (bp_name, bp) in &blueprints {
        let bp = bp.clone();
        println!(
            "\n## blueprint: {bp_name}: exploitability {:.4}, round-2-only exploitability {:.4}",
            exploitability(&Leduc, &bp),
            round2_exploitability(&bp, &bp)
        );
        for ((sname, sf), play_average) in solver_sets.iter().flat_map(|s| [(s, false), (s, true)]) {
            let play = if play_average { "plays the weighted average (diagnostic, not the paper)" } else { "plays the final iteration (paper)" };
            println!("\n### solvers: {sname}; {play}");
            println!("| agent | exploitability mean [min, max] | round-2-only exploitability mean [min, max] | vs blueprint mean [min, max] | searches |");
            println!("|---|---|---|---|---|");
            for (name, n_cont, r0, r1) in variants {
                let (mut ex, mut ex2, mut hh, mut ns) = (Vec::new(), Vec::new(), Vec::new(), 0);
                for seed in 1..=3u64 {
                    let (early, late) = sf(seed);
                    let cfg = SearchConfig { n_cont, compress: if std::env::args().any(|a| a == "--no-compress") { None } else { Some(1) }, search_round0: r0, search_round1: r1, early, late, no_reraise: false, play_average, follow_continuation: name.starts_with("F") };
                    let mut ag = Agent::new(&bp, cfg);
                    let (pol, owner) = agent_policy(&mut ag);
                    ex.push(exploitability(&Leduc, &pol));
                    ex2.push(round2_exploitability(&pol, &bp));
                    hh.push(head_to_head(&pol, &owner, &bp));
                    ns = ag.searches;
                }
                let f = |v: &[f64]| {
                    let mean = v.iter().sum::<f64>() / v.len() as f64;
                    let (lo, hi) = v.iter().fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
                    format!("{mean:.4} [{lo:.4}, {hi:.4}]")
                };
                println!("| {name} | {} | {} | {} | {ns} |", f(&ex), f(&ex2), f(&hh));
            }
        }
    }
    eprintln!("total {:.1}s", t0.elapsed().as_secs_f64());
}
