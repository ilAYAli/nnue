All development is done on pwa-llm ~/code/chess/nnue
read AGENTS.md there and the associated SKILL.md's

## Current champion

enyo-1.32.0-rc10 (continue_from enyo-1.31.0-rc57), commit 650a4fcc (2026-07-29).
Absolute vs the fixed external target nn-0ee0657fb25e.nnue:
elo=-139.8  ci=19.2  (4000 games, 2026-07-30, engine enyo_11ca4d7)
This is the number to beat in absolute terms; promotion itself is decided parent-relative.

Also recorded, own separate Elo scale, not directly comparable to the nn-0ee0657fb25e.nnue
numbers below: enyo-1.30.0-rc3 vs berserk-9b84c340af7e.nn: elo=-122.2 (1500 games, ci=31.9).

All Elo figures below are vs nn-0ee0657fb25e.nnue unless stated otherwise. The "vs
default.net" progression that used to live here is dropped from this document -
default.net is a much weaker, non-comparable reference and added confusion rather than
signal. See benchmarks/default-net.jsonl directly if that history is ever needed.

## Traceable ancestry (continue_from-verified)

### Scratch origin -> enyo-1.0.0-rc1 (2026-06-18 to 2026-07-07)

  commit    run                           continue_from                  parent-rel result
  2947c36   enyo-scratch-long-1.0.0-rc1   (scratch, no continue_from)    -
  5a6028e   "        "  (same commit)     "        "                     +93.2  (2.26/2.20, 103%) vs enyo-scratch-broad-1.5.0-rc1, 1500g
  7c7a076   enyo-scratch-long-1.1.0-rc1   enyo-scratch-long-1.0.0-rc1    +45.9  (2.12/2.20, 96%)
  a1c8368   enyo-scratch-long-1.2.0-rc1   enyo-scratch-long-1.1.0-rc1    +25.8  (1.17/2.20, 53%)
  1c22805   enyo-scratch-long-1.3.0-rc1   enyo-scratch-long-1.2.0-rc1    +16.0  (0.71/2.20, 32%)
  d6c42e4   enyo-scratch-long-1.4.0-rc1   enyo-scratch-long-1.3.0-rc1    +7.4   (0.35/2.20, 16%)
  8cea17a   enyo-scratch-long-1.5.0-rc1   enyo-scratch-long-1.4.0-rc1    +0.5   (0.02/2.20, 1%)
  9d829ac   enyo-1.0.0-rc1                enyo-scratch-long-1.5.0-rc1    +4.9   (0.18/2.20, 8%)

  None of this chain except its final step was ever benchmarked directly vs
  nn-0ee0657fb25e.nnue:
  enyo-1.0.0-rc1 vs SF: elo=-181.2  ci=29.3  (1000 games, 2026-07-07) - the true
  starting point of the numbered lineage's absolute distance from Stockfish.

### enyo-1.1.0 through enyo-1.32.0, excluding the traceable rc3/rc57/rc10 milestones below: no recoverable continue_from chain (2026-07-07 to 2026-07-15)

  A high-churn architecture/hyperparameter search phase - well over a hundred rc
  candidates across ~30 promotion numbers in 9 days, no parent links preserved. Every
  directly-recorded absolute result vs nn-0ee0657fb25e.nnue in this stretch, in date
  order (source: benchmarks/stockfish-net.jsonl; duplicate rows are repeat/re-runs of
  the same candidate, kept as-is for a faithful record):

  2026-07-07  enyo-1.1.0-rc1               elo= -173.2  games=1000  ci=24.5
  2026-07-07  enyo-1.1.0-rc2               elo= -185.8  games=1000  ci=27.6
  2026-07-07  enyo-1.2.0-rc1               elo= -184.4  games=1000  ci=30.2
  2026-07-07  enyo-1.5.0-rc2               elo= -148.0  games=500   ci=28.5
  2026-07-08  enyo-1.10.0-rc4              elo= -187.1  games=500   ci=28.9
  2026-07-08  enyo-1.11.0-rc2              elo= -197.4  games=500   ci=29.8
  2026-07-08  enyo-1.3.0-rc2               elo= -180.8  games=500   ci=27.9
  2026-07-08  enyo-1.7.0-rc3               elo= -148.0  games=500   ci=27.6
  2026-07-08  enyo-1.7.0-rc3               elo= -158.9  games=500   ci=28.4
  2026-07-08  enyo-1.9.0-rc1               elo= -180.8  games=500   ci=28.4
  2026-07-09  enyo-1.12.0-rc1              elo= -176.3  games=500   ci=28.8
  2026-07-09  enyo-1.13.0-rc1              elo= -179.0  games=500   ci=28.7
  2026-07-09  enyo-1.13.0-rc2              elo= -158.9  games=500   ci=28.5
  2026-07-09  enyo-1.14.0-rc1              elo= -178.1  games=500   ci=28.1
  2026-07-09  enyo-1.14.0-rc2              elo= -179.9  games=500   ci=28.0
  2026-07-09  enyo-1.15.0-rc1              elo= -188.1  games=500   ci=28.6
  2026-07-09  enyo-1.15.0-rc2              elo= -171.9  games=500   ci=26.2
  2026-07-09  enyo-1.16.0-rc1              elo= -190.8  games=500   ci=27.8
  2026-07-09  enyo-1.16.0-rc2              elo= -183.5  games=500   ci=28.7
  2026-07-09  enyo-1.16.0-rc3              elo= -184.4  games=500   ci=28.4
  2026-07-09  enyo-1.16.0-rc3              elo= -150.5  games=500   ci=27.8
  2026-07-09  enyo-1.17.0-rc1              elo= -156.4  games=500   ci=27.9
  2026-07-09  enyo-1.17.0-rc2              elo= -158.1  games=500   ci=26.1
  2026-07-09  enyo-1.18.0-rc1              elo= -155.5  games=500   ci=28.8
  2026-07-09  enyo-1.18.0-rc1              elo= -156.4  games=500   ci=28.9
  2026-07-09  enyo-1.19.0-rc1              elo= -187.1  games=500   ci=30.3
  2026-07-09  enyo-1.19.0-rc2              elo= -172.8  games=500   ci=28.3
  2026-07-09  enyo-1.19.0-rc2              elo= -168.4  games=500   ci=30.5
  2026-07-09  enyo-1.20.0-rc1              elo= -175.4  games=500   ci=27.9
  2026-07-09  enyo-1.20.0-rc2              elo= -183.5  games=500   ci=31.9
  2026-07-09  enyo-1.20.0-rc3              elo= -183.5  games=500   ci=29.0
  2026-07-09  enyo-1.5.0-rc2               elo= -171.9  games=500   ci=28.5
  2026-07-09  enyo-1.5.0-rc2               elo= -160.6  games=500   ci=26.4
  2026-07-10  enyo-1.20.0-rc10             elo= -188.1  games=500   ci=30.6
  2026-07-10  enyo-1.20.0-rc11             elo= -201.2  games=500   ci=31.4
  2026-07-10  enyo-1.20.0-rc12             elo= -168.4  games=500   ci=28.1
  2026-07-10  enyo-1.20.0-rc4              elo= -186.2  games=500   ci=30.1
  2026-07-10  enyo-1.20.0-rc5              elo= -169.3  games=500   ci=27.4
  2026-07-10  enyo-1.20.0-rc6              elo= -195.5  games=500   ci=30.3
  2026-07-10  enyo-1.20.0-rc7              elo= -164.9  games=500   ci=27.1
  2026-07-10  enyo-1.20.0-rc8              elo= -173.7  games=500   ci=29.0
  2026-07-10  enyo-1.20.0-rc9              elo= -166.7  games=500   ci=28.8
  2026-07-10  enyo-1.21.0-rc3              elo= -173.6  games=500   ci=29.7
  2026-07-10  enyo-1.21.0-rc5              elo= -173.7  games=500   ci=29.4
  2026-07-10  enyo-1.21.0-rc7              elo= -179.0  games=500   ci=29.4
  2026-07-10  enyo-1.21.0-rc8              elo= -158.9  games=500   ci=28.3
  2026-07-10  enyo-1.21.0-rc9              elo= -174.5  games=500   ci=28.2
  2026-07-10  enyo-1.22.0-rc1              elo= -180.8  games=500   ci=28.5
  2026-07-10  enyo-1.22.0-rc2              elo= -186.2  games=500   ci=30.2
  2026-07-11  enyo-1.16.0-rc3              elo= -170.1  games=2000  ci=13.1
  2026-07-11  enyo-1.20.0-rc12             elo= -161.9  games=2000  ci=13.1
  2026-07-11  enyo-1.21.0-rc8              elo= -182.8  games=2000  ci=13.8
  2026-07-11  enyo-1.22.0-rc10             elo= -178.1  games=500   ci=28.7
  2026-07-11  enyo-1.22.0-rc3              elo= -182.6  games=500   ci=28.0
  2026-07-11  enyo-1.22.0-rc4              elo= -178.1  games=500   ci=28.6
  2026-07-11  enyo-1.22.0-rc5              elo= -164.1  games=500   ci=28.1
  2026-07-11  enyo-1.22.0-rc6              elo= -173.7  games=500   ci=28.2
  2026-07-11  enyo-1.22.0-rc7              elo= -171.0  games=500   ci=28.8
  2026-07-11  enyo-1.22.0-rc8              elo= -168.4  games=500   ci=28.6
  2026-07-11  enyo-1.22.0-rc9              elo= -171.0  games=500   ci=29.1
  2026-07-11  enyo-1.23.0-rc2              elo= -171.0  games=500   ci=29.0
  2026-07-11  enyo-1.24.0-rc1              elo= -183.5  games=500   ci=28.4
  2026-07-11  enyo-1.25.0-rc2              elo= -164.9  games=500   ci=27.9
  2026-07-11  enyo-1.25.0-rc3              elo= -171.9  games=500   ci=29.5
  2026-07-11  enyo-1.26.0-rc1              elo= -170.1  games=500   ci=28.8
  2026-07-12  enyo-1.27.0-rc1              elo= -201.2  games=500   ci=32.8
  2026-07-12  enyo-1.28.0-rc1              elo= -179.0  games=500   ci=27.8
  2026-07-12  enyo-1.28.0-rc10             elo= -170.1  games=500   ci=28.3
  2026-07-12  enyo-1.28.0-rc11             elo= -154.7  games=500   ci=28.5
  2026-07-12  enyo-1.28.0-rc12             elo= -173.7  games=500   ci=28.1
  2026-07-12  enyo-1.28.0-rc13             elo= -168.4  games=500   ci=28.6
  2026-07-12  enyo-1.28.0-rc14             elo= -189.0  games=500   ci=29.7
  2026-07-12  enyo-1.28.0-rc15             elo= -189.0  games=500   ci=27.9
  2026-07-12  enyo-1.28.0-rc16             elo= -158.1  games=500   ci=25.8
  2026-07-12  enyo-1.28.0-rc19             elo= -168.4  games=500   ci=26.9
  2026-07-12  enyo-1.28.0-rc2              elo= -178.1  games=500   ci=29.2
  2026-07-12  enyo-1.28.0-rc20             elo= -172.8  games=500   ci=28.3
  2026-07-12  enyo-1.28.0-rc3              elo= -158.9  games=500   ci=28.8
  2026-07-12  enyo-1.28.0-rc4              elo= -162.3  games=500   ci=27.1
  2026-07-12  enyo-1.28.0-rc5              elo= -171.0  games=500   ci=28.4
  2026-07-12  enyo-1.28.0-rc6              elo= -158.9  games=500   ci=28.1
  2026-07-12  enyo-1.28.0-rc7              elo= -175.4  games=500   ci=28.5
  2026-07-12  enyo-1.28.0-rc8              elo= -188.1  games=500   ci=30.9
  2026-07-12  enyo-1.28.0-rc9              elo= -179.0  games=500   ci=28.9
  2026-07-12  enyo-1.29.0-rc1              elo= -161.5  games=500   ci=27.3
  2026-07-12  enyo-1.29.0-rc2              elo= -189.9  games=500   ci=27.9
  2026-07-13  enyo-1.28.0-rc21             elo= -173.6  games=500   ci=28.0
  2026-07-13  enyo-1.28.0-rc22             elo= -152.2  games=500   ci=27.6
  2026-07-13  enyo-1.28.0-rc23             elo= -196.4  games=500   ci=30.4
  2026-07-13  enyo-1.28.0-rc24             elo= -181.7  games=500   ci=29.6
  2026-07-13  enyo-1.28.0-rc25             elo= -179.9  games=500   ci=27.9
  2026-07-13  enyo-1.28.0-rc26             elo= -177.2  games=500   ci=28.1
  2026-07-13  enyo-1.28.0-rc27             elo= -166.7  games=500   ci=27.7
  2026-07-13  enyo-1.28.0-rc28             elo= -183.5  games=500   ci=29.7
  2026-07-13  enyo-1.28.0-rc29             elo= -171.0  games=500   ci=27.0
  2026-07-13  enyo-1.28.0-rc30             elo= -181.7  games=500   ci=29.8
  2026-07-13  enyo-1.28.0-rc31             elo= -178.1  games=500   ci=30.3
  2026-07-13  enyo-1.28.0-rc32             elo= -168.4  games=500   ci=29.8
  2026-07-13  enyo-1.28.0-rc33             elo= -178.1  games=500   ci=28.5
  2026-07-13  enyo-1.28.0-rc37             elo= -188.1  games=500   ci=31.9
  2026-07-13  enyo-1.28.0-rc38             elo= -173.7  games=500   ci=28.8
  2026-07-13  enyo-1.28.0-rc39             elo= -192.7  games=500   ci=29.2
  2026-07-13  enyo-1.28.0-rc40             elo= -169.3  games=500   ci=28.2
  2026-07-13  enyo-1.28.0-rc41             elo= -175.4  games=500   ci=28.0
  2026-07-13  enyo-1.28.0-rc42             elo= -162.4  games=500   ci=28.7
  2026-07-13  enyo-1.28.0-rc43             elo= -192.7  games=500   ci=27.7
  2026-07-13  enyo-1.28.0-rc44             elo= -178.1  games=500   ci=28.1
  2026-07-13  enyo-1.28.0-rc45             elo= -184.4  games=500   ci=27.7
  2026-07-14  enyo-1.31.0-rc20             elo= -185.3  games=500   ci=31.6
  2026-07-14  enyo-1.31.0-rc22             elo= -177.2  games=500   ci=28.4
  2026-07-14  enyo-1.31.0-rc24             elo= -148.8  games=500   ci=26.7
  2026-07-14  enyo-1.31.0-rc26             elo= -168.4  games=500   ci=26.6
  2026-07-14  enyo-1.31.0-rc28             elo= -194.6  games=500   ci=29.0
  2026-07-14  enyo-1.31.0-rc30             elo= -197.4  games=500   ci=29.8
  2026-07-14  enyo-1.31.0-rc32             elo= -169.3  games=500   ci=29.6
  2026-07-14  enyo-1.31.0-rc36             elo= -179.0  games=500   ci=28.1
  2026-07-14  enyo-1.31.0-rc38             elo= -164.9  games=500   ci=27.3
  2026-07-14  enyo-1.31.0-rc39             elo= -180.8  games=500   ci=28.5
  2026-07-14  enyo-1.31.0-rc40             elo= -187.2  games=500   ci=28.9
  2026-07-14  enyo-1.31.0-rc41             elo= -182.6  games=500   ci=28.4
  2026-07-14  enyo-1.31.0-rc42             elo= -155.5  games=500   ci=28.8
  2026-07-14  enyo-1.32.0-rc2              elo= -185.3  games=500   ci=29.0
  2026-07-14  enyo-1.32.0-rc4              elo= -200.2  games=500   ci=30.9
  2026-07-14  enyo-1.32.0-rc5              elo= -194.6  games=500   ci=29.9
  2026-07-14  enyo-1.32.0-rc6              elo= -202.1  games=500   ci=30.5
  2026-07-15  enyo-1.32.0-rc11             elo= -210.9  games=500   ci=32.8
  2026-07-15  enyo-1.32.0-rc12             elo= -159.8  games=500   ci=27.6
  2026-07-15  enyo-1.32.0-rc13             elo= -157.2  games=500   ci=28.0
  2026-07-15  enyo-1.32.0-rc14             elo= -158.1  games=500   ci=28.0
  2026-07-15  enyo-1.32.0-rc15             elo= -188.1  games=500   ci=29.3
  2026-07-15  enyo-1.32.0-rc16             elo= -188.1  games=500   ci=27.7
  2026-07-15  enyo-1.32.0-rc17             elo= -158.1  games=500   ci=27.2
  2026-07-15  enyo-1.32.0-rc7              elo= -196.4  games=500   ci=28.7
  2026-07-15  enyo-1.32.0-rc8              elo= -199.3  games=500   ci=31.1
  2026-07-15  enyo-1.32.0-rc9              elo= -208.9  games=500   ci=30.7

  Best result in this whole untraced stretch: enyo-1.17.0-rc1, elo=-156.4 (500 games,
  2026-07-09) - but with no continue_from proof, "best" here just means "lowest measured
  absolute gap," not a verified lineage peak.

  Also benchmarked in this window but off the enyo-1.x line entirely (rejected
  architecture probes, all far worse than anything on the main line):

  2026-07-10  enyo-2.0.0-rc1               elo= -559.2  games=500   ci=253.3
  2026-07-10  enyo-2.0.0-rc2               elo= -426.6  games=500   ci=76.1
  2026-07-11  enyo-2.0.0-rc3               elo= -354.5  games=500   ci=51.7
  2026-07-11  enyo-2.0.0-rc4               elo= -465.4  games=500   ci=142.3
  2026-07-12  enyo-2.0.0-rc5               elo= -254.1  games=500   ci=32.7
  2026-07-12  enyo-2.0.0-rc6               elo= -265.7  games=500   ci=34.0
  2026-07-12  enyo-2.0.0-rc7               elo= -281.7  games=500   ci=39.1
  2026-07-13  enyo-3.0.0-rc1               elo= -194.6  games=500   ci=28.2
  2026-07-13  enyo-4.0.0-rc1               elo= -185.3  games=500   ci=28.9
  2026-07-13  enyo-arch-control-1.0.0-rc1  elo= -259.9  games=500   ci=36.4

### enyo-1.30.0-rc3 forward to current champion (verified 2026-08-02)

  enyo-1.30.0-rc3 itself is untraceable further back: no recoverable commit anywhere in
  history (checked all commits' build.json content, not just subject-line grep) and its
  net file no longer exists on disk. Versions 1.1.0-1.29.0 above were not traced since
  there is no endpoint to connect them to this chain.

  commit    run                           continue_from                parent-rel   vs SF
  -         enyo-1.30.0-rc3               (untraceable)                -            -154.4  (1500g, 07-17)
  -         enyo-1.30.0-rc3-unscaled      enyo-1.30.0-rc3 (manual fix) n/a          -
  3c464a3   enyo-1.31.0-rc57              enyo-1.30.0-rc3-unscaled     +8.6         -146.7  (4000g, corrected)
  650a4fc   enyo-1.32.0-rc10 (champion)   enyo-1.31.0-rc57             +3.3         -139.8  (4000g, corrected)

  Both rc57 and rc10's vs-SF numbers were corrected on 2026-07-30 from earlier
  smaller-sample/engine-mismatch readings (-145.5 @ 1008g and -183.5/-183.5 @ 500g
  respectively) - see the "correction" note further down for why.

  Naming collision: the name enyo-1.32.0-rc10 was used twice. An earlier, unrelated
  attempt was rejected on 2026-07-15 (commit 7e31912f, continue_from enyo-1.31.0-rc42,
  elo=-69.6 parent-relative) - a different net that happened to reuse the same name.
  Only commit 650a4fcc (2026-07-29, above) is the actual current champion file.

## Rejected side-branches off the current tip (2026-08-01/02 onward, this session):

  enyo-1.33.0-rc9  (continue_from rc10, data/selfplay/gen4/gen4-sf-oracle.bullet - self-play
                    generated with Stockfish's own net loaded into the search engine, i.e. the
                    wrong policy net for self-distillation): elo=-17.3  llr=-2.40/2.20 (-109%)
                    (1708/6000 games) - rejected.
  enyo-1.33.0-rc10 (continue_from rc10, data/selfplay/gen5/gen5-sf-oracle.bullet - corrected
                    recipe, rc10 plays itself, genuine SF static-eval relabel, matching rc57's
                    actual recipe): elo=-9.0  llr=-2.25/2.20 (-102%) (3014/6000 games) - rejected.
  Conclusion: even the corrected self-play recipe does not reproduce rc57's gain against the
  now-mature rc10 - the incremental self-play-fine-tune path off rc10 appears saturated.

*** 2026-07-30 reconstruction ***

enyo-ancestor.1.0.0-rc1: reconstructed enyo-scratch-long-1.0.0-rc1's architecture/regimen from scratch
(196608 superbatches, wdl 0.05, min_ply 24, no continue_from) since the original
enyo-scratch-broad-1.0.0-rc1.bullet corpus no longer exists and its provenance is untraceable.
Substituted lc0-static + nodes5000pv2-recalibrated, both reconverted/relabeled at min_ply 24.
Result vs SF: elo=-208.6  llr=-19.24/690.78 (-3%)  los=0.0%  (1500 games, 2026-07-30).
~27 Elo below enyo-1.0.0-rc1's -181.2, same ballpark - the historical regimen isn't broken, the gap
is plausibly the data substitution. See benchmarks/stockfish-net.jsonl and commit e4e32ea.

Separately, tonight's own fresh 2-round scratch bootstrap (enyo-1.33.0-rc3 -> rc4, min_ply 16, same
combined corpus, 76762 superbatches/round, continuing from the prior round's own checkpoint each time)
reached elo=-180.6 vs SF after just 2 rounds (~153524 superbatches total) - matching enyo-1.0.0-rc1's
-181.2 starting point almost exactly, with LESS total dose than the single 196608-superbatch
enyo-ancestor.1.0.0-rc1 run above, which landed worse (-208.6). This suggests multiple shorter rounds
(fresh lr schedule reset each round) generalizes better than one long monolithic run of equal-or-greater
total dose - consistent with how the real historical chain was structured (7 successive rounds, not one).
Both rc3 and rc4 were rejected as promotion candidates against enyo-1.32.0-rc10 (the bar is much higher
now than enyo-1.0.0-rc1's original starting point), which is expected and not evidence against the
approach itself.

*** orphaned candidates (2026-07-30) ***

enyo-1.33.0-rc5.nn: trained and exported (continue_from rc10, lr 1e-5, farseerT74-recalibrated-600m.bullet,
4000 superbatches) but never benchmarked. A retroactive SPRT vs SF was started and then deliberately stopped
before completion - farseerT74-recalibrated-600m.bullet uses the same calibration curve already flagged
as likely mis-calibrated for this source, so spending a full 1500-game run on a presumed-bad candidate
wasn't worth it. No Elo vs SF exists for this candidate.

enyo-1.33.0-rc6: continue_from rc4, 196608 superbatches on lc0-nodes5000pv2-2b.bullet (min_ply 16) - started,
stopped almost immediately (only checkpoints/native-0 exists, no export) before the pivot to the
enyo-ancestor.1.0.0-rc1 reconstruction work. No net was ever produced, so no Elo vs SF exists for this
candidate.

*** correction (2026-07-30, later same day) ***

The rc57-vs-rc10 poison investigation above was resolved by clean re-benchmarks: both nets tested vs SF
with matched conditions (current engine enyo_11ca4d7, 4000 games each, tight bounds elo0=0/elo1=10/
alpha=beta=1e-300):

enyo-1.31.0-rc57.nn: elo=-146.7  ci=21.0  (4000/4000 games)
enyo-1.32.0-rc10.nn: elo=-139.8  ci=19.2  (4000/4000 games)

Statistically indistinguishable, rc10 if anything very slightly ahead. The original ~38 Elo gap
(-145.5 partial/invalid vs -183.5 at 500 games/older engine) that suggested rc10 had regressed from its
own parent was a measurement artifact - small sample size and an engine-binary mismatch between the two
original readings, not a real continue_from-chain regression. rc10 remains a perfectly good current
lineage tip; no re-basing needed. The best net vs SF currently on record is rc10/rc57, both ~-140 to -147.

*** self-play relabeling consistency investigation (2026-07-31 overnight) ***

Investigated whether data/selfplay/gen1/gen2/gen3 use consistent labeling before combining them.
Found they don't: gen1-live-sfo/gen1-sf-oracle.bullet (rc57's winning data) is SF-oracle static-eval
only, no tablebase correction. gen2.bullet and gen3/selfplay-sf-oracle-tb.bullet are both SF-oracle +
TB-corrected (manifest for gen2.bullet traces to a run literally named tb-sf-oracle-relabel-...).

Ran enyo-1.33.0-rc8 (gen3's non-TB variant selfplay-sf-oracle.bullet, continue_from rc10, same regimen
as the rejected rc7) as a single-variable isolation test. Result: elo=-2.7 ci=7.6 los=24.3% (6000/6000
games, full sample, no early termination) - not promoted (los well below the 75% threshold), but a large
improvement over rc7's clear rejection (elo=-23.1, los=6.1%). Consistent with (not proof of) tablebase
correction distorting calibration - see the rc7 residual_gate note above about static-metrics-improve-
but-play-doesn't.

Attempted to build a Forge template (selfplay-to-bullet-sf-oracle.template.json, in the forge repo,
NOT committed) to distribute tools/bullet/selfplay_to_bullet_sf_oracle.py across the fleet for
relabeling gen2 consistently (gen2 is the one generation with raw PGNs still on disk - gen1/gen3's
raw games were already cleaned up after their original conversion). Verified the underlying script
and its exact default parameters (skip_plies=8, min_depth=1, max_abs_cp=10000, mode=static,
tb_pieces=0, sf_net=nn-0ee0657fb25e.nnue) work correctly on a small local (non-distributed) sample.

The distributed Forge deployment of the new template hung: a 20-PGN-file test run
(selfplay-to-bullet-sf-oracle-.tmp-sfo-template-test-20260731-012513) sat at 0/20 done for ~3.7 hours
before I caught it and force-stopped it (forge stop --force: released=7 killed=48 reset=7). No task
logs were written at all, meaning tasks likely never got past setup (venv/cargo build install steps)
rather than hanging mid-relabel - root cause NOT diagnosed. Did not attempt a second blind distributed
run given the fleet-time cost of the first attempt (~4 hours across dozens of workers).

Template file is left in place on pwa-llm at
~/code/chess/forge/templates/selfplay-to-bullet-sf-oracle.template.json, uncommitted, NOT working.
Test artifacts left for inspection: ~/code/chess/nnue/data/selfplay/.tmp-sfo-template-test/ (20 PGN
files) and the Forge run directory
~/code/chess/forge/runs/selfplay-to-bullet-sf-oracle-.tmp-sfo-template-test-20260731-012513/.

Next step if picking this back up: debug the template's setup phase specifically (materialize_inputs,
the nnue source install step, or the bullet-utils cargo build) on a single worker directly before
attempting another fleet-wide deploy - the hang happened before any task produced output, so it's a
setup/deploy problem, not a labeling-script problem (the script itself is confirmed correct locally).

*** update: root cause partially found (2026-07-31, later) ***

Found and fixed one real bug in selfplay-to-bullet-sf-oracle.template.json: the nnue_repo include list
was missing tools/score/** (relabel_with_stockfish.py imports label_with_uci from there), causing
ModuleNotFoundError on every worker since only the included subset gets materialized - confirmed via
manual single-host reproduction (ssh + exact task command), fixed, and manually re-verified working
end-to-end on one worker (2156 positions converted and validated correctly, ~470 pos/sec).

Redeploying via Forge after the fix still did not complete a 20-PGN-file test (0/20 done after several
minutes). Diagnosed further: the PGN input directory was never materialized on the worker I checked
(~/.cache/forge/inputs/selfplay/ did not contain this test's cache key, despite the run showing tasks
as claimed/running). This is the materialize_inputs mechanism itself (core Forge behaviour, not
something a template JSON can fix) failing to complete/sync for this template, even for a handful of
small files. Stopped again (forge stop --force) rather than keep debugging Forge core solo - that's
outside what a template-only fix can address and is the other Claude session's remit (Forge core).

Net state: the template's own bug (missing include) is fixed and should be preserved if anyone picks
this back up, but there's a second, deeper issue in the input-materialization path that needs either
Forge-side debugging or the user's input before attempting gen2's real relabeling job again. Do not
retry the full job blind - both attempts so far cost real fleet time (~4h then ~15min) for zero
completed output.

*** enyo-scc-1.0.0-rc1 (2026-08-05): new lineage, rejected ***

Full Lc0 test91 conversion completed (30,174-shard distributed Forge label job,
1.906B raw records -> 1,232,772,104 selected positions, quiet-only/min_ply=16).
Combined with nodes5000pv2-recalibrated-2b.bullet (2,000,000,000 positions) into
a 3,232,772,104-position corpus, interleaved via bullet-utils interleave (a
genuine random cross-source merge - the `direct` loader has no shuffle buffer
of its own, just a 256MB read-ahead one, so the corpus itself must already be
shuffled), and scratch-trained (no continue_from, 9 epochs, wdl=0.05 - the
enyo-scratch-long-1.0.0-rc1 regimen) as a new, separate lineage
(enyo-scc-1.0.0-rc1), SPRT'd directly against champion enyo-1.32.0-rc10.

Result: elo=-93.2  llr=-32.60/690.78 (-5%)  draw=45.5%  (4000/4000 games) -
rejected, as expected for a single-generation scratch lineage against a
champion that has been through dozens of fine-tune iterations (compare
enyo-1.0.0-rc1's own -181.2 vs SF starting point, above). Not evidence against
the Lc0 data itself - the champion comparison isn't the relevant signal for
judging a founding net's data quality, only for the promotion decision.

enyo-scc-1.0.0-rc1 vs nn-0ee0657fb25e.nnue (absolute, documentation only):
elo=-136.2  llr=-11.76/690.78 (-2%)  draw=45.2%  (1500/1500 games, 2026-08-05).
Meaningfully better than enyo-ancestor.1.0.0-rc1's -208.6 (same two data
sources, unshuffled concatenation) and close to enyo-1.0.0-rc1's own -181.2
founding-net starting point - a reasonable place for a first-generation
lineage to land.

*** enyo-scc-1.1.0-rc1 (2026-08-05): dose-continuation round 1, rejected ***

continue_from enyo-scc-1.0.0-rc1 (rejected, elo=-93.2), same regimen/corpus
enriched with a one-time blend of existing, already-validated self-play data
(gen1/gen3/gen5 sf-oracle, non-TB, right policy net - 136,088,579 positions,
~4% growth), 231321 superbatches (scaled with the corpus growth), wdl=0.05 -
mirroring how enyo-scratch-long got 5 rounds of real gains via dose
continuation alone.

Result: elo=-66.9  llr=-26.24/690.78 (-4%)  draw=46.1%  (4000/4000 games) -
still rejected vs champion enyo-1.32.0-rc10 (expected), but +26.3 Elo over
rc1.0 on the same reference - a real per-round gain, in the range of
enyo-scratch-long's own early rounds (+93.2, +45.9, +25.8...). Validates
dose-continuation as the ongoing lever for this lineage.

*** enyo-scc-1.1.0-rc2 (2026-08-06): shuffle retry, residual gate rejection ***

Retry of rc1.1 from the same parent (enyo-scc-1.0.0-rc1) after an additional
bullet-utils shuffle pass on the combined corpus, to test whether residual
same-source clustering from the interleave step (invisible to the sequential
direct loader) explained the -238.7 vs-SF measurement on rc1.1.

Result: residual gate rejection - never reached SPRT. The gate measures
slope_gain (candidate improvement over champion in eval-vs-reference slope,
per phase and eval bucket). rc2 failed two checks: endgame slope_gain=-0.004
and eval:800+ slope_gain=-0.001, both below MIN_SLOPE=0.05. MAE improved
substantially in both groups (endgame: 313.8→227.8) but slope did not, meaning
the net is more accurate in absolute terms but less well-calibrated vs the
champion. The margin of failure is tiny and likely within residual-audit noise
(50k samples, comparable to the CI on the slope estimate). Shuffle hypothesis
is inconclusive - rc2 is not demonstrably better or worse than rc1.1.

*** enyo-scc-1.2.0-rc1 (2026-08-06): training objective experiment, rejected ***

continue_from enyo-scc-1.1.0-rc1 with three simultaneous changes: WDL 0.05→0.3
(primary hypothesis - game-result weight never tested above 0.05 in this
lineage), cubic loss |e|^3 replacing squared loss (spike_trainer rebuild,
+14.8 Elo at fixed nodes per July 2024 research), AdamW beta1 0.9→0.95 (+4
Elo per same research), final_lr 5e-6→2.5e-5 (fix 200:1 LR decay ratio).

vs nn-0ee0657fb25e.nnue: elo=-223.9  llr=-13.60/690.78 (-2%)  ci=13.9
draw=48.0%  (1500/1500 games, 2026-08-06) - catastrophic. 87 Elo worse than the
founding net (-136.2) and ~110 Elo below rc1.1's estimated absolute.

Most likely cause: WDL=0.3 with the mixed corpus. The Lc0 T91 positions (38%
of corpus) have game results from MCTS outcomes but eval labels from SF oracle
static eval. At WDL=0.05 these are nearly decoupled. At WDL=0.3, the MCTS
game outcomes become a dominant training signal and conflict with SF oracle
evals for positions where Lc0 tactically outplayed SF from a neutral static
eval. The net is pulled in two directions simultaneously.

Next: enyo-scc-1.2.0-rc2 - revert WDL to 0.05, keep cubic loss and beta1=0.95
as a baseline improvement bundle. This isolates the code changes from the WDL
hypothesis.
