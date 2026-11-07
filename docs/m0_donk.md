# M0: donk-bet audit

Definition (strict): on street s, player X **called the last bet/raise** of street s-1 (made by someone else), X is **first to act** on street s, and X **bets**. Denominator = all such spots. Computed for every player in the 10,000 hands.

## Rates by street

| street | Pluribus opps | Pluribus donks | Pluribus rate | humans opps | humans donks | humans rate |
|---|---:|---:|---:|---:|---:|---:|
| flop | 577 | 9 | 1.6% | 3,018 | 28 | 0.9% |
| turn | 159 | 18 | 11.3% | 894 | 28 | 3.1% |
| river | 88 | 2 | 2.3% | 388 | 15 | 3.9% |
| **all** | 824 | 29 | 3.5% | 4,300 | 71 | 1.7% |

## Heads-up vs multiway (flop, Pluribus)

| players who called into the flop (incl. aggressor) | opps | donk rate |
|---|---:|---:|
| 2 | 482 | 0.8% |
| 3 | 90 | 5.6% |
| 4 | 5 | 0.0% |

## Why the earlier figure was 2.1% (v1 in m0_stats.md)

v1 flop spots: 612 (donks 13, 2.1%). Strict flop spots: 577 (donks 9, 1.6%).

- In v1 but not strict: 35 hands
- In strict but not v1: 0 hands
- In both: 577 hands

Of the 35 v1-only spots, 35 had another player check before Pluribus on the flop (so Pluribus was not first to act). v1 also counted only the flop. Under both definitions Pluribus' flop donk rate is low; its donking is concentrated on the turn.

## 10 sample Pluribus spots: donk bets

| hand | street | Pluribus | aggressor | board | previous-street line | bet to |
|---|---|---|---|---|---|---:|
| 106/209 | turn | SB | CO | 2d4cKh4s | SB:cc BB:cc CO:cbr 225 SB:cc BB:f | 281 |
| 113b/160 | turn | BB | BTN | 8hAsAd7h | BB:cc BTN:cbr 200 BB:cc | 237 |
| 30/41 | flop | SB | BTN | 4h9s6h | UTG:f MP:f CO:f BTN:cbr 250 SB:cc BB:cc | 750 |
| 40/76 | turn | BB | BTN | 6c7hJcJh | BB:cc BTN:cbr 312 BB:cc | 312 |
| 40b/219 | turn | UTG | CO | 9h2sTh8h | UTG:cc CO:cbr 200 UTG:cc | 250 |
| 44b/149 | flop | SB | MP | 4hTdAs | UTG:f MP:cbr 225 CO:f BTN:f SB:cc BB:cc | 675 |
| 50b/184 | flop | BB | MP | 8dAh7h | UTG:f MP:cbr 225 CO:f BTN:cc SB:f BB:cc | 725 |
| 85/58 | turn | BB | BTN | Qd5h4s3h | BB:cc BTN:cbr 275 BB:cc | 275 |
| 96b/124 | turn | BB | MP | Qs2c5c5h | BB:cc MP:cbr 150 BB:cc | 1140 |
| 97b/92 | turn | MP | BTN | 2s3sQs4h | MP:cc BTN:cbr 555 MP:cc | 665 |

## 10 sample Pluribus spots: checks (no donk)

| hand | street | Pluribus | aggressor | board | previous-street line | bet to |
|---|---|---|---|---|---|---:|
| 108/190 | flop | BB | CO | 5c4hQc | UTG:f MP:f CO:cbr 225 BTN:f SB:f BB:cc |  |
| 111/53 | flop | SB | BTN | TcQc4h | UTG:f MP:f CO:f BTN:cbr 225 SB:cc BB:f |  |
| 111b/94 | flop | BB | BTN | 8hAc9s | UTG:f MP:f CO:f BTN:cbr 275 SB:f BB:cc |  |
| 117/35 | flop | SB | BTN | 4c4sJd | UTG:f MP:f CO:f BTN:cbr 250 SB:cc BB:cc |  |
| 40/107 | flop | SB | CO | 2s3h8h | UTG:f MP:f CO:cbr 225 BTN:f SB:cc BB:cc |  |
| 44b/110 | turn | MP | BTN | Js3h3c6s | MP:cc BTN:cbr 150 MP:cc |  |
| 61/122 | turn | MP | CO | 5hTc4s8c | MP:cc CO:cbr 225 MP:cc |  |
| 63/160 | turn | BB | UTG | 7sKsAhTs | BB:cc UTG:cbr 250 BB:cc |  |
| 75b/28 | flop | BB | BTN | Qc4sTc | UTG:f MP:f CO:f BTN:cbr 225 SB:f BB:cc |  |
| 98/97 | river | CO | BTN | Tc9cJc6dQc | BB:cc CO:cc BTN:cbr 180 BB:f CO:cc |  |
