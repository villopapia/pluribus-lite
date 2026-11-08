# Sources review (Phase A2): what the extended sources say about our open items

Written 2026-11-08. **Recommendations only: no code was changed.** Each needs your approval
before it becomes a change; accepted ones go into `docs/deviations.md`.

**Evidence levels** (strongest first):
1. **paper**: Pluribus main paper or supplement
2. **cited method**: a paper Pluribus cites for that component, or the authors' predecessor
   system (Libratus, Baby Tartanian8)
3. **author statement**: blog, press release, AMA, talks
4. **our inference**: our own reasoning or measurement

## 1. Sources retrieved

PDFs are in `docs/papers/` and web pages in `docs/sources/`. Both are gitignored, because they
are third-party texts we do not redistribute. Retrieved 2026-11-08. "Ref" is the number in the
Pluribus supplement's reference list.

That list gives authors, venue and pages but **no titles**. Every title below was verified
against the first page of the fetched file and matched to the list by authors, venue, year and
page range.

| ref | verified title | file | source URL | SHA-256 | level |
|---|---|---|---|---|---|
| — (Libratus) | Brown & Sandholm, *Superhuman AI for heads-up no-limit poker: Libratus beats top professionals*, Science 359:418–424 (2018) | `Brown2018-Science-Libratus.pdf` | noambrown.github.io/papers/17-Science-Superhuman.pdf | 7c8dc72bf3d6953bf1366e2b7b3f6c211b6621cd7b58b50f065a309c9df6217a | cited method (predecessor) |
| — (Libratus Supp) | Brown & Sandholm, *Supplementary Materials for Superhuman AI for heads-up no-limit poker: Libratus beats top professionals*, Science, DOI 10.1126/science.aao1733 (the DOI is printed in the Libratus paper) | `Brown2018-Science-Libratus_Supp.pdf` (38 pages) | science.org/doi/suppl/10.1126/science.aao1733/suppl_file/aao1733_brown_sm.pdf (downloaded by the user, 2026-11-08) | fc42e46153fe89ee2f372198a5064f0260e37590f337a16df1da67991cea209b | cited method (predecessor) |
| 23 | Gibson, *Regret Minimization in Games and the Development of Champion Multiplayer Computer Poker-Playing Agents*, PhD thesis, U. Alberta 2014 | `Gibson2014-PhD-thesis.pdf` | ualberta.scholaris.ca/bitstreams/c770eae9-e78d-4c17-91f1-d71076c244ab/download | 30fdddc7d9bc3ab7fb1e2a1337870bf4628d48d43fdd757b7e5b7be38600c3a1 | cited method |
| 27 | Ganzfried & Sandholm, *Potential-Aware Imperfect-Recall Abstraction with Earth Mover's Distance in Imperfect-Information Games*, AAAI 2014, pp. 682–690 | `Ganzfried2014-AAAI-potential-aware-EMD.pdf` | cs.cmu.edu/~sandholm/potential-aware_imperfect-recall.aaai14.pdf | 27b87c8d0bf4be8e8b3a7efc3f825c3bb2a1624fd834748726e0df46f5bc0ed7 | cited method |
| 30 | Jackson, *Slumbot NL: Solving Large Games with Counterfactual Regret Minimization Using Sampling and Distributed Processing*, AAAI Workshop on Computer Poker and Imperfect Information 2013 | `Jackson2013-AAAIW-slumbot-nl.pdf` | cdn.aaai.org/ocs/ws/ws0979/7044-30516-1-PB.pdf | 0c12ad002b80ba5b798060ed9de085c20f516e2c16fa57365cd10efb00e6f242 | cited method |
| 32 | Jackson, *Compact CFR*, AAAI Workshop on Computer Poker and Imperfect Information Games 2016 (WS-16-06) | `Jackson2016-AAAIW-compact-cfr.pdf` | cdn.aaai.org/ocs/ws/ws0231/12647-57457-1-PB.pdf (link found on aaai.org/proceeding/ws16-06/ via Chrome) | 62fe785a45eb0a0a611fa952cd173a5e736aba8defa86407be2b45eb870f8fa0 | cited method |
| 34 | Jackson, *Targeted CFR*, AAAI Workshop on Computer Poker and Imperfect Information Games 2017 | `Jackson2017-AAAIW-targeted-cfr.pdf` | cdn.aaai.org/ocs/ws/ws0407/15219-68366-1-PB.pdf | 3b7842257247c9a84bac563096295827140016abc535905b2e728b8a2ef65961 | cited method |
| 33 | Brown & Sandholm, *Baby Tartanian8: Winning Agent from the 2016 Annual Computer Poker Competition*, IJCAI-16, pp. 4238–4239 | `Brown2016-IJCAI-baby-tartanian8.pdf` | ijcai.org/Proceedings/16/Papers/639.pdf | 7b8cb9286993819cb550ea53f057a6f6acc3130f02c29df2c9753094e67443cf | cited method |
| 35 | Lanctot, Waugh, Zinkevich, Bowling, *Monte Carlo Sampling for Regret Minimization in Extensive Games*, NIPS 2009, pp. 1078–1086 | `Lanctot2009-NIPS-MCCFR.pdf` | mlanctot.info/files/papers/nips09mccfr.pdf | 80bbcdd3b2743de4094682d08af54674a9ca03a5505b2d4b7ba666540b950f53 | cited method |
| 42 | Johanson, Waugh, Bowling, Zinkevich, *Accelerating Best Response Calculation in Large Extensive Games*, IJCAI 2011, pp. 258–265 | `Johanson2011-IJCAI-accelerating-BR.pdf` | ijcai.org/Proceedings/11/Papers/054.pdf | fec9d8a338dead6debb54c4d62b8037e9ce26580d4c61d14daa118bbb11f7ff9 | cited method |
| 47 | Waugh, *A Fast and Optimal Hand Isomorphism Algorithm*, AAAI Workshop on Computer Poker and Imperfect Information 2013 | `Waugh2013-AAAIW-hand-isomorphism.pdf` | cs.cmu.edu/~kwaugh/publications/isomorphism13.pdf | 6954faf16606ca3eee92c6d506f8e192cf0b30581e21516c4ea70ea993b53911 | cited method |
| 48 | Gilpin, Sandholm, Sørensen, *Potential-Aware Automated Abstraction of Sequential Games, and Holistic Equilibrium Analysis of Texas Hold'em Poker*, AAAI 2007, pp. 50–57 | `Gilpin2007-AAAI-potential-aware.pdf` | cdn.aaai.org/AAAI/2007/AAAI07-007.pdf | 6d433dd356ee63bbf76f27df9672190e1d6a1453db2211e7ba83a3a553738a68 | cited method |
| 49 | Ganzfried & Sandholm, *Endgame Solving in Large Imperfect-Information Games*, AAMAS 2015, pp. 37–45 | `Ganzfried2015-AAMAS-endgame.pdf` | ifaamas.org/Proceedings/aamas2015/aamas/p37.pdf | 056405f7f17f34b87c2807062b5e2943051637e4eb0098c48c3e512a7564ef39 | cited method |
| 52 | Brown & Sandholm, *Safe and Nested Subgame Solving for Imperfect-Information Games*, NeurIPS 2017, pp. 689–699 | `Brown2017-NeurIPS-safe-nested-subgame.pdf` | arxiv.org/pdf/1705.02955 | f70b5cf6be537a6b010052a4c6b3581ed4777e0194a7a4548b82f4c88a62fbfb | cited method |
| — | Brown & Sandholm, *Regret-Based Pruning in Extensive-Form Games*, NIPS 2015 (with appendix) | `Brown2015-NeurIPS-regret-based-pruning.pdf` | cs.cmu.edu/~sandholm/regret-basedPruning.nips15.withAppendix.pdf | 8a24de4ba21e4cf10392d5599109cfe08eb1f0523858601abc98f6fe8c4c5f3f | cited method (pruning lineage) |
| — | Brown & Sandholm, *Reduced Space and Faster Convergence in Imperfect-Information Games via Pruning*, ICML 2017 | `Brown2017-ICML-pruning.pdf` | arxiv.org/pdf/1609.03234 | 828bed1513f1732169a5e9fd7e7fb85dc4a2dfcae72742562dec0ee5579e8381 | cited method (pruning lineage) |
| — | Meta AI blog, *Facebook, Carnegie Mellon build first AI that beats pros in 6-player poker*, 11 Jul 2019 | `docs/sources/meta_ai_blog_pluribus_2019-07-11.md` | ai.meta.com/blog/pluribus-first-ai-to-beat-pros-in-6-player-poker/ | fcd109a5bd5c131c900e135ba75af09bdcdbf5c3e93ee6e77c23f7d2c93f0204 (28,799 bytes; hashes the Dell's converted `.md`, not the live page). Archive: web.archive.org/web/20230925033116/https://ai.meta.com/blog/pluribus-first-ai-to-beat-pros-in-6-player-poker/ (earliest snapshot of this URL; no 2019 snapshot of the original ai.facebook.com address was found) | author statement |
| — | CMU News, *Carnegie Mellon and Facebook AI Beats Professionals in Six-Player Poker*, 11 Jul 2019 | `docs/sources/cmu_news_pluribus_2019-07-11.md` | cmu.edu/news/stories/archives/2019/july/cmu-facebook-ai-beats-poker-pros.html | 536254ade9e91b033dadef51de3bd45dbec156f8ef1038103924934d7f9290fb (9,455 bytes; hashes the Dell's converted `.md`, not the live page). Archive: web.archive.org/web/20190712211632/https://www.cmu.edu/news/stories/archives/2019/july/cmu-facebook-ai-beats-poker-pros.html | author statement (press release) |

**Not retrieved.** Each needs a manual download. Please save the file into `docs/papers/` or
`docs/sources/`, as you did for Data S1.

| item | why | title as known |
|---|---|---|
| AMA, r/MachineLearning, 17–19 Jul 2019 (thread `ceece3`) | the fetch tool refuses reddit.com; the one mirror returned HTTP 502 | *AMA: We are Noam Brown and Tuomas Sandholm, creators of the Carnegie Mellon / Facebook multiplayer poker bot Pluribus* |

### What the Libratus supplement adds (read in full, 2026-11-08)

- **No bet-size sets and no card-abstraction details** are published in it either. Size choice:
  sizes come from analysing ACPC agents and from parameter optimisation (main paper); overnight
  "self-improvement" adds the most common opponent sizes, or gap-filling ones, scored by
  min{x − A, B − x}.
- **Pruning (its Algorithm 1)** differs from Pluribus':
  - after T₀ iterations, *each* traverser action with regret below C is sampled with probability
    K / (K + C − R), with a floor of 2% on the chained sampling probability;
  - sampled actions update φ with weight 1/p;
  - there are **no exceptions** for the final round or for terminal actions.

  Pluribus' rule (95% / 5% per iteration, exceptions for the final round and terminal actions)
  is a different, simpler variant. The Libratus text therefore gives **no evidence on how to
  read Pluribus' exceptions** (D-ALG-9).
- **φ in Libratus:** it is incremented at **opponent** nodes on every iteration, from t = 1, on
  all rounds (standard external sampling). Pluribus changed this to a sampled traverser update
  every 10,000 iterations, round 1 only (Alg. 1). This confirms that Pluribus' φ rule is a
  deliberate change, not an inherited default.
- **Search (lineage only, for M4):**
  - CFR+ with 1,000 iterations if at most $1,600 is in the pot, otherwise 1,900;
  - all bet sizes randomly scaled by 0–8% per hand (footnote 49);
  - perfect-recall abstraction in subgames.
- **Hardware:** 196 nodes for the blueprint; only 14 of 28 cores used per node ("bottlenecked
  primarily by memory access"); about 3M core hours for the blueprints.

## 2. Technical details from author statements that are not in the supplement

| detail | quote (short) | source | level | status vs the supplement |
|---|---|---|---|---|
| Limping was in the blueprint's action space, and training removed it | "The graphic also notes when Pluribus stops 'limping'" | Meta blog | author statement | new; **supports `limp = all` in `faithful.cfg` (D-ACT-8)** |
| Blueprint progress was measured against the final snapshot, without search | "Performance is measured against the final snapshot of training. We do not use search in these comparisons." | Meta blog | author statement | new; an **evaluation protocol** we can copy |
| Multiple preflop raise sizes in play | pro quote: "Unlike humans, Pluribus used multiple raise sizes preflop." | Meta blog | author statement (second hand) | consistent with Supp p. 12 ("1 to 14") |
| Training compute | "eight days on a 64-core server … less than 512 GB of RAM. No GPUs" | Meta blog | author statement | same as the paper |
| Core hours | "12,400 core hours … just 28 cores during live play" | CMU | author statement | the paper has 12,400; 28 cores = the paper's two 14-core E5-2695 v3, so "two CPUs" (Meta) is consistent |
| Leaf continuation strategies | Meta: "four different strategies"; CMU: "five possible continuation strategies" | Meta / CMU | author statement | the **paper says four** (Supp pp. 23–24); the CMU count is a press-release discrepancy. Keep four |
| Search time | "between one second and 33 seconds … 20 seconds per hand" | Meta blog | author statement | same as the paper |
| No opponent adaptation | "Pluribus does not adapt its strategy to the observed tendencies of its opponents." | Meta blog | author statement | consistent |
| Code lineage | "Pluribus builds on and incorporates large parts of that technology and code" (Strategic Machine / CMU lab) | CMU | author statement | explains why unpublished details follow the Libratus / Baby Tartanian8 lineage |

No author statement gives bet-size sets, bucket features, iteration counts, search iteration
budgets or the solver-switch threshold.

## 3. Open items in `deviations.md`: what the sources say, and what we recommend

| item | what the new sources say | level | recommendation |
|---|---|---|---|
| **D-ACT-1 / 2 / 3** (preflop/flop size sets, depths without evidence) | The **Libratus supplement** (now read) publishes no sets either. **Baby Tartanian8**: "Actions were selected by examining the equilibrium strategies of smaller agents and choosing the actions that were most commonly used." **Libratus**: sizes were "nice fractions or multiples of the pot [roughly determined by analyzing the most common bet sizes … taken by prior top AIs in the ACPC]"; a few early sizes came from a parameter-optimisation algorithm (Brown & Sandholm AAAI 2014, not fetched). Neither gives actual sets; the Libratus supplement may (not retrieved). | cited method | **Keep the method** (frequent sizes on a round-fraction grid, calibrated to the tree size). It matches the lineage's stated practice, and D-ACT-5 shows Pluribus' raises sit on round fractions. No change. |
| **D-ACT-6** (abstract size below the min-raise: dropped, not clamped) | nothing in any source | — | Keep "dropped" (a clamped size would duplicate the min-raise or another size). No change. |
| **D-ACT-7** (a short all-in reopens betting) | Nothing in the sources. **Our inference:** every hand starts with equal 10,000 stacks (Supp p. 4), so any all-in is to the common maximum; no player can re-raise above it, and reopening can never be used. | our inference (logical) | **Close as "no effect"** after one confirming test (on the faithful tree, no raise follows an all-in). Code change: a test only. |
| **D-ACT-8** (limping allowed) | the Meta blog: the blueprint learned to stop limping during training | author statement | Supports the current `limp = all`. Upgrade the evidence in the ledger; no code change. |
| **D-INFO-1** (flop/turn features; **one k-means run**) | **Ganzfried & Sandholm 2014** (ref 27): "we conducted **25 restarts** using the k-means++ initialization procedure … and selected the run that produced the **lowest within-cluster sum of squares**". Johanson 2013 ([26]) also uses restarts. Potential-aware EMD ([27], [48], [28]) is cited **only for the search abstraction**; for the blueprint the paper cites only [26]. Libratus / Tartanian7 used a hierarchical abstraction (public flop buckets, then private buckets). | cited method | **Adopt restarts** for the faithful abstraction build: k-means++ with **25 restarts**, keep the lowest within-cluster sum of squares (mean EMD for flop/turn, squared L2 for river). Cost is small: one fit is ~6–7 s on the sample, so ~3 min per street. Keep hand-strength histograms for the blueprint, and use potential-aware EMD only for M4 search, as the paper does. *Code change: `build_abstraction --restarts 25`; rebuild `j200` and `j63`. The Dell run uses `j63`, so apply this at the next fresh start, not mid-run.* |
| **D-INFO-2** (river OCHS) | nothing new; [27] notes "there is no potential on the river" | cited method | Keep OCHS. No change. |
| **D-INFO-5** (direct (hole, board) indexing without suit isomorphism) | Waugh 2013 (ref 47) is cited only for the **lossless** round-1 abstraction (Supp p. 13); our 169 preflop classes are exactly that | cited method | No change. Note in the ledger that [47] concerns round 1 only. |
| **D-SCHED-1** (paper minute as a fraction of the planned run) | nothing new (Meta repeats 8 days, 64 cores) | — | Keep. |
| **D-ALG-5** (φ from t = 0; ablation at 800) | **Gibson thesis** (ref 23), multiplayer CFR (Alg., lines 32–34): the cumulative profile is updated only with probability p on some iterations, a precursor of Pluribus' strategy interval; nothing on a start time | cited method | Keep the ruling. Run the planned ablation (0 vs 800) when there is a reduced config to spare. No change. |
| **D-ALG-6** (post-flop snapshots: uniform average, missing = uniform) **(resolved 2026-11-08: Compact CFR read; it uses an equally weighted average = ours)** | The paper cites **Jackson 2016 (Compact CFR) [32]** for averaging snapshots; its full text is not retrieved, and the abstract does not mention averaging weights. **Gibson thesis**: in 3-player limit hold'em, the **current** profile reached better performance than the average (5 vs 20 days, Fig. 4.7b), which supports averaging current-strategy snapshots after the flop. | cited method (partly unread) | Keep uniform averaging. **Retrieve Compact CFR manually**: if it weights snapshots (e.g. linearly), we should match it. No change until then. |
| **D-ALG-9** (pruning exceptions) | Pruning lineage: Regret-Based Pruning (NIPS 2015) skips an action only until its regret could turn positive; **Baby Tartanian8** (ref 33) used a sampled RBP variant, "turned off RBS for the final 5 days" because of "some evidence that our implementation … may have led to decreased performance when closer to convergence". No source discusses fold or run-out exemptions. The **Libratus supplement's** pruning is per-action probabilistic (K / (K + C − R), 2% floor) with **no** exceptions, so Pluribus' exception rule is its own and the lineage cannot settle our reading. Jackson 2017 (Targeted CFR, ref 34) is not retrieved. | cited method | Keep the 2026-11-08 ruling. Watch `pruned_rate` and convergence late in the run. The Baby Tartanian8 note is a reason to keep the 5% unpruned iterations exactly as in the paper. No change. |
| **D-ALG-10** (parallelism: lock-free threads) | Baby Tartanian8 / Tartanian7: distributed MCCFR, flop-conditioned partitions; Pluribus used one shared-memory node (paper) | cited method | No change (shared memory matches the paper's hardware). |
| **D-RES-4** (63 buckets on the Dell) | — | — | No change. |
| **D-INFO-6 / M4 search** (pending) | **Johanson 2011** (ref 42): the vector-based traversal ("public chance sampling": sample public cards, carry vectors over all private hands) is the CFR form Pluribus' search uses with one board sample per thread. **Brown & Sandholm 2017** (ref 52): nested subgame solving and its action-translation handling. **Ganzfried & Sandholm 2015** (ref 49): unsafe "endgame solving", the variant Pluribus adopts | cited method | Fold into the M4 design. Write `docs/m4_design.md` from these before coding M4. No change now. |
| **D-EVAL-1** (AIVAT) | Meta: "a version of the AIVAT variance reduction algorithm"; no further detail | author statement | No change. Add the Meta protocol to `eval_plan.md` (blueprint checkpoints vs the final snapshot, no search) as a cheap progress curve; this is a docs change only. |

## 4. Summary of proposed changes (awaiting your review)

1. **D-INFO-1:** k-means++ with 25 restarts (lowest within-cluster SS) for the abstraction
   build; rebuild abstractions at the next fresh start. *Code and data.*
2. **D-ACT-7:** add a test proving no raise can follow an all-in under equal stacks; then close
   the item. *Test only.*
3. **D-ACT-8, D-INFO-5:** upgrade the evidence in the ledger. *Docs only.*
4. **Eval plan:** add the Meta-style blueprint progress curve. *Docs only.*
5. **Manual download needed:** the Reddit AMA thread (save the page as HTML or PDF into
   `docs/sources/`). The Jackson papers were retrieved on 2026-11-08.
