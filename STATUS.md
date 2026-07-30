All development is done on pwa-5090 ~/code/cpp/chess/nnue
read AGENTS.md there and the associated SKILL.md's

The best net is currently ~ -154 elo from SF and ~ -122 from Berserk:

enyo-1.30.0-rc3.nn vs berserk-9b84c340af7e.nn (1500 games):
elo=-122.2  llr=-2.29/2.20 (-104%)  los=0.0%  ci=31.9  draw=35.4%

enyo-1.30.0-rc3.nn vs nn-0ee0657fb25e.nnue (1500 games):
elo=-154.4  llr=-22.66/690.78 (-3%)  los=0.0%  ci=15.4  draw=32.9%


Update: the currently best net is always ~/assts/nets/reference.net (enyo-1.32.0-rc10.nn)

The NNUE architecture is failry close to Berserk, but I have been able to close the gap further.
I have tried different architectures, selfplay, stockfish binpacks, and additional features such
as FullThreats.

Look at the git log to investigate the different lineages.

I am contemplating re-doing an earlier experiment and have a competition between multiple architecutes and/or features, trained without using existing tensors.


*** This is the enyo-1 lineage ***

commit: 9d829ac0
"run:" enyo-1.0.0-rc1
"continue_from": enyo-scratch-long-1.5.0-rc1

commit: 8cea17a7
"run": "enyo-scratch-long-1.5.0-rc1",
"continue_from": "enyo-scratch-long-1.4.0-rc1"

commit: d6c42e4c
"run": "enyo-scratch-long-1.4.0-rc1"
"continue_from": "enyo-scratch-long-1.3.0-rc1"

commit: 1c22805e
"enyo-scratch-long-1.3.0-rc1"
"continue_from": "enyo-scratch-long-1.2.0-rc1"

commit: a1c8368a
"run": "enyo-scratch-long-1.2.0-rc1"
"continue_from": "enyo-scratch-long-1.1.0-rc1"

commit: 7c7a0764
"run": "enyo-scratch-long-1.1.0-rc1",
"continue_from": "enyo-scratch-long-1.0.0-rc1"

commit: 5a6028ef
"run": "enyo-scratch-long-1.0.0-rc1"


*** default.net reference ***

default.net (used below) is ~/code/rice/src/hexadecane_512_v2.net on pwa-mbp (with 4 bytes appended, IIRC) -
a DIFFERENT reference engine than nn-0ee0657fb25e.nnue used everywhere else in this file. Elo numbers
against the two are not directly comparable.

*** enyo-1 lineage elo progression (vs default.net, benchmarks/default-net.jsonl) ***

enyo-scratch-broad-1.0.0-rc1   -236.0  (1000 games, 2026-07-03)
enyo-scratch-long-1.0.0-rc1    -169.3  (500 games,  2026-07-05)
enyo-scratch-long-1.1.0-rc1    -126.2  (500 games,  2026-07-05)
enyo-scratch-long-1.3.0-rc1    -113.7  (500 games,  2026-07-06)
(1.2.0, 1.4.0, 1.5.0 not present in benchmarks/default-net.jsonl)

enyo-1.0.0-rc1 (continue_from enyo-scratch-long-1.5.0-rc1) was the first net of this chain ever
benchmarked against Stockfish directly:
elo=-181.2  llr=-17.52/690.78  los=0.0%  ci=29.3  (1000 games, 2026-07-07, benchmarks/stockfish-net.jsonl:1)

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
