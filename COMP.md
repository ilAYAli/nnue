# Architecture competition results

The scratch results below are baseline measurements, not the final architecture
ranking. The competitive cluster (`h4`, `h8`, `h16`, and `sf32`) advances with
both seeds into the same matched continuation regimen. The clearly weaker
`h1`, `h16w768`, and `h16o4` controls do not advance. Architecture selection
uses the refined two-seed results; it does not select the best isolated scratch
score.

## Long continuation: `h8` seed 1 (`enyo-h8-v3-rc1`)

`enyo-h8-v3-rc1.nn` continued `enyo-h8-v2-rc1.nn` for 7,600 superbatches on
the matched disjoint slice. SHA-256:
`c15aaac4a9a7b1ac7648c6d901fd17ce1cd6563ed3ce01ceaa2e6e76f54ab4e1`.
Static evaluation was `mae=138.153`, `sign=84.03%`, `corr=0.874558`,
`bias=-0.242`, and `slope=0.823114`. Residual MAE improved for endgame
`420.366 -> 412.136` and eval 800+ `643.374 -> 625.768`, but regressed for
eval 300-799 `350.209 -> 350.921`.

Status: rejected by `residual_gate`; no SPRT was run. The regression is small
(`-0.713` MAE gain) but violates the fixed gate, so this checkpoint supplies no
game-strength evidence.

## Long continuation: `h4` seed 1 (`enyo-h4-v3-rc1`)

`enyo-h4-v3-rc1.nn` continues `enyo-h4-v2-rc1.nn` for 7,600 superbatches on
the next disjoint data slice. SHA-256:
`df02792c3faf8f1f21796ff2be8b59f90e9108ccc647d259c5103e82247f87a1`.
Gates passed with static `mae=136.116`, `sign=84.01%`, `corr=0.873809`,
`bias=6.904`, and `slope=0.814035`. Residual MAE improved for endgame
`428.738 -> 424.590`, eval 800+ `652.761 -> 646.411`, and eval 300-799
`353.810 -> 349.234`.

enyo-h4-v3-rc1-sprt-1500-20260722-144424 22 Jul 14:44
"candidate=enyo-h4-v3-rc1.nn vs reference=enyo-h4-v2-rc1.nn"
elo=+4.2  llr=0.20/2.20 (9%)  los=71.0%  ci=14.8  draw=37.6%  games=1500/1500  tasks=88/88

The long continuation is inconclusive and shows a weak marginal learning rate.
Compare the first long seed of every lineage before deciding whether `h4`
warrants its second long seed.

## Matched continuation: `sf32` seed 1 (`enyo-sf32-v2-rc1`)

`enyo-sf32-v2-rc1.nn` applies the matched 256-superbatch continuation to
`enyo-sf32-v1-rc1.nn`. SHA-256:
`2b5f9aca41fa5df6594cb3f66a241de82fdd0a4fc84a11592d6a415f6f67cef9`.
Gates passed with static `mae=136.372`, `sign=83.16%`, `corr=0.860505`,
`bias=2.863`, and `slope=0.793384`. Residual MAE improved for endgame
`457.526 -> 415.116`, eval 800+ `682.312 -> 634.993`, and eval 300-799
`381.473 -> 347.194`.

enyo-sf32-v2-rc1-sprt-1500-20260722-125307 22 Jul 12:53
"candidate=enyo-sf32-v2-rc1.nn vs reference=enyo-sf32-v1-rc1.nn"
elo=+15.3  llr=0.67/2.20 (30%)  los=97.9%  ci=14.7  draw=38.4%  games=1500/1500  tasks=88/88

This is a positive first-seed refinement signal. Complete the matched second
seed before comparing the `sf32` learning trajectory with the other lineages.

## Matched continuation: `sf32` seed 2 (`enyo-sf32-v2-rc2`)

`enyo-sf32-v2-rc2.nn` applies the identical continuation to
`enyo-sf32-v1-rc2.nn`. SHA-256:
`7beaa45a0e47fa13e681517ba6e5882de0537ed4dc3d5ae9c63b1accdf316720`.
Gates passed with static `mae=138.326`, `sign=82.60%`, `corr=0.861523`,
`bias=3.556`, and `slope=0.803956`. Residual MAE improved for endgame
`461.610 -> 419.533`, eval 800+ `689.476 -> 644.096`, and eval 300-799
`388.997 -> 354.893`.

enyo-sf32-v2-rc2-sprt-1500-20260722-132928 22 Jul 13:29
"candidate=enyo-sf32-v2-rc2.nn vs reference=enyo-sf32-v1-rc2.nn"
elo=+14.8  llr=0.71/2.20 (32%)  los=97.8%  ci=14.4  draw=40.3%  games=1500/1500  tasks=88/88

Across both seeds, the matched `sf32` continuation gain is approximately
`+15.1 +/-10.3` Elo. It is statistically indistinguishable from `h4` and `h8`
at this depth and numerically below `h16`.

## Matched continuation: `h16` seed 1 (`enyo-h16-v2-rc1`)

`enyo-h16-v2-rc1.nn` applies the matched 256-superbatch continuation to
`enyo-h16-v1-rc1.nn`. SHA-256:
`d792eb683cf96409f37d61ce75d0b3c6635158b38332dfbf20d600d642945f31`.
Gates passed with static `mae=134.376`, `sign=83.88%`, `corr=0.870559`,
`bias=3.539`, and `slope=0.798668`. Residual MAE improved for endgame
`458.345 -> 409.651`, eval 800+ `680.086 -> 626.343`, and eval 300-799
`392.206 -> 352.789`.

enyo-h16-v2-rc1-sprt-1500-20260722-113608 22 Jul 11:36
"candidate=enyo-h16-v2-rc1.nn vs reference=enyo-h16-v1-rc1.nn"
elo=+25.3  llr=1.15/2.20 (52%)  los=100.0%  ci=14.7  draw=38.6%  games=1500/1500  tasks=88/88

This is the largest first-seed gain so far, but its confidence interval overlaps
the `h4` and `h8` results. Await the matched second seed.

## Matched continuation: `h16` seed 2 (`enyo-h16-v2-rc2`)

`enyo-h16-v2-rc2.nn` applies the identical continuation to
`enyo-h16-v1-rc2.nn`. SHA-256:
`44da0e11c3c2d4ab07a204299ee2561d3e3fa39c51f97444eecfff1e2951833c`.
Gates passed with static `mae=136.275`, `sign=83.60%`, `corr=0.865994`,
`bias=-0.493`, and `slope=0.794574`. Residual MAE improved for endgame
`441.404 -> 411.199`, eval 800+ `663.214 -> 631.183`, and eval 300-799
`390.174 -> 359.073`.

enyo-h16-v2-rc2-sprt-1500-20260722-121338 22 Jul 12:13
"candidate=enyo-h16-v2-rc2.nn vs reference=enyo-h16-v1-rc2.nn"
elo=+20.6  llr=1.01/2.20 (46%)  los=99.7%  ci=14.5  draw=39.7%  games=1500/1500  tasks=88/88

Across both seeds, the matched `h16` continuation gain is approximately
`+23.0 +/-10.3` Elo. This is numerically above `h4` and `h8`, but the lineage
means still overlap substantially.

## Matched continuation: `h8` seed 1 (`enyo-h8-v2-rc1`)

`enyo-h8-v2-rc1.nn` applies the matched 256-superbatch continuation to
`enyo-h8-v1-rc1.nn`. SHA-256:
`c37f5492ac279a094ca3f2531518360fb4784e01c880b580fa8be929c7a13168`.
Gates passed with static `mae=136.068`, `sign=83.94%`, `corr=0.867272`,
`bias=-0.987`, and `slope=0.800557`. Residual MAE improved for endgame
`463.751 -> 420.366`, eval 800+ `692.500 -> 643.374`, and eval 300-799
`387.137 -> 350.209`.

enyo-h8-v2-rc1-sprt-1500-20260722-102300 22 Jul 10:23
"candidate=enyo-h8-v2-rc1.nn vs reference=enyo-h8-v1-rc1.nn"
elo=+22.5  llr=1.04/2.20 (47%)  los=99.9%  ci=14.3  draw=40.5%  games=1500/1500  tasks=88/88

This is a positive first-seed refinement signal. It does not select `h8` until
the matched second seed and the remaining lineages are measured.

## Matched continuation: `h8` seed 2 (`enyo-h8-v2-rc2`)

`enyo-h8-v2-rc2.nn` applies the identical continuation to
`enyo-h8-v1-rc2.nn`. SHA-256:
`95e54bcff5f6ca1feb483fc3ddb5ca632dd601fb1301f07a9f5480ded10d2480`.
Gates passed with static `mae=135.992`, `sign=83.39%`, `corr=0.866509`,
`bias=-0.103`, and `slope=0.786493`. Residual MAE improved for endgame
`445.692 -> 408.135`, eval 800+ `658.759 -> 617.356`, and eval 300-799
`363.931 -> 332.552`.

enyo-h8-v2-rc2-sprt-1500-20260722-105938 22 Jul 10:59
"candidate=enyo-h8-v2-rc2.nn vs reference=enyo-h8-v1-rc2.nn"
elo=+10.7  llr=0.47/2.20 (21%)  los=92.0%  ci=14.8  draw=36.9%  games=1500/1500  tasks=88/88

Across both seeds, the matched `h8` continuation gain is approximately
`+16.6 +/-10.3` Elo. It is statistically indistinguishable from the current
`h4` mean of `+15.1 +/-10.4` Elo.

## Matched continuation: `h4` seed 1 (`enyo-h4-v2-rc1`)

`enyo-h4-v2-rc1.nn` continues `enyo-h4-v1-rc1.nn` for 256 superbatches on
the first 200M T60T70/Farseer positions with `lr=0.0001`, `wdl=0.05`, and
`activation_l1=0.00001`. SHA-256:
`daedb7b9265fb6eb26e05fbbb88ad5aea42accc14e3ce6d79fdd7cd4c5ba71a9`.

Gates passed: start position `+48cp`; static `mae=138.493`, `sign=83.62%`,
`corr=0.863823`, `bias=2.286`, `slope=0.800089`. Residual MAE improved in all
required bands: endgame `470.484 -> 428.738`, eval 800+ `702.237 -> 652.761`,
and eval 300-799 `390.252 -> 353.810`. The optional move gate was skipped
because its cases file was absent and strict mode was disabled.

enyo-h4-v2-rc1-sprt-1500-20260722-090106 22 Jul 09:01
"candidate=enyo-h4-v2-rc1.nn vs reference=enyo-h4-v1-rc1.nn"
elo=+21.3  llr=0.93/2.20 (42%)  los=99.8%  ci=14.5  draw=39.2%  games=1500/1500  tasks=88/88

The same-regimen continuation is a positive local refinement signal. The
separate mature-reference run was intentionally cancelled at 388/1500 games
to avoid spending matches that can be reserved for the final leading lineages;
it is invalid for ranking and receives no benchmark result.

## Matched continuation: `h4` seed 2 (`enyo-h4-v2-rc2`)

`enyo-h4-v2-rc2.nn` applies the identical continuation to
`enyo-h4-v1-rc2.nn`. SHA-256:
`ed64efe813c7ebb9e7271dd414b20ec2fe810aa0b6775aa4f7b0961327dc2423`.
Gates passed with static `mae=136.365`, `sign=83.63%`, `corr=0.864225`,
`bias=-1.784`, and `slope=0.799118`. Residual MAE improved for endgame
`459.057 -> 418.856`, eval 800+ `685.284 -> 641.567`, and eval 300-799
`386.169 -> 354.467`.

enyo-h4-v2-rc2-sprt-1500-20260722-094515 22 Jul 09:45
"candidate=enyo-h4-v2-rc2.nn vs reference=enyo-h4-v1-rc2.nn"
elo=+8.8  llr=0.38/2.20 (17%)  los=87.7%  ci=14.9  draw=36.7%  games=1500/1500  tasks=88/88

Across both seeds, the matched `h4` continuation gain is approximately
`+15.1 +/-10.4` Elo. This is a replicated positive refinement signal, not an
architecture-selection result; the other lineages receive the same treatment.

## J: current Reckless 10x12-768-o8 with threats (`recklessft`)

Net: `enyo-recklessft-v1-rc1.nn`

The first export transposed both dense layers incorrectly. Its static result and
`0-1499-1` game score are invalid and retained only as exporter-bug provenance.
After fixing the dense layout, the preserved trained checkpoint was re-exported
without additional training. The corrected net has SHA-256
`93acbbcec2b67dea234effe57a4f51c5b7a65084d541e734436854be866a835c`.

The corrected 50,000-position runtime evaluation was sane but failed the
residual gate: `mae=348.059120`, `sign=75.427673%`, `corr=0.723222`,
`bias=-27.901760`, and `slope=2.024792`. Endgame residual MAE regressed from
`285.109` to `396.507`, eval `800+` from `432.990` to `508.697`, and eval
`300-799` from `164.106` to `402.053`.

enyo-recklessft-v1-rc1-sprt-1500-20260722-081235 22 Jul 08:12 · 20m
"candidate=enyo-recklessft-v1-rc1.nn vs reference=enyo-1.30.0-rc3.nn"
elo=-919.54  llr=-13.11/2.20  los=0.0%  ci=93.68  draw=0.9%  games=1500/1500  tasks=88/88

The all-shard score was `1-1486-13` (wins-losses-draws), with zero task
failures. Elo, CI, LOS, and draw rate are reconstructed from every shard. The
LLR is the sum of the final per-shard LLRs at their available two-decimal
precision. Forge's finite-shard aggregate (`-572.5` Elo, `-1.73/2.20` LLR,
`6.2%` draws) omitted the 76 all-loss shards and is invalid. This corrected
result rejects the trained `recklessft` candidate; do not train rc2.

## H: 16x12-1024-o4, factorised (`h16o4`)

Net: `enyo-h16o4-v1-rc2.nn`

Training and export completed. The residual gate rejected the candidate:
endgame residual MAE changed from `285.109` to `434.041`, the `800+` bucket
changed from `432.990` to `646.066`, and the `300-799` bucket changed from
`164.106` to `371.302`. This is a calibration gate result, not an Elo result;
the replicate remains unranked until its game test is recorded.

sprt-enyo-h16o4-v1-rc2.nn-vs-reference.net-1000-20260721-090832 21 Jul 09:08 · 16m
"candidate=enyo-h16o4-v1-rc2.nn vs reference=reference.net"
elo=-156.1  llr=-2.22/2.20 (-101%)  los=0.0%  ci=36.2  draw=33.9%  games=292/1000  tasks=24/24

The two-seed mean is `-148.4 Elo` versus `reference.net` (`rc1=-140.7`,
`rc2=-156.1`). Both replicates were decisively rejected by the SPRT used for
these anchor measurements.

Net: `enyo-h16o4-v1-rc1.nn`

sprt-enyo-h16o4-v1-rc1.nn-vs-reference.net-1000-20260721-074355 21 Jul 07:43 · 8m
"candidate=enyo-h16o4-v1-rc1.nn vs reference=reference.net"
elo=-140.7  llr=-2.29/2.20 (-104%)  los=0.0%  ci=37.7  draw=35.4%  games=328/1000  tasks=24/24

## G: 16x12-768-o8, factorised (`h16w768`)

Net: `enyo-h16w768-v1-rc2.nn`

sprt-enyo-h16w768-v1-rc2.nn-vs-reference.net-1000-20260720-212922 20 Jul 21:29 · 10m
"candidate=enyo-h16w768-v1-rc2.nn vs reference=reference.net"
elo=-126.4  llr=-2.44/2.20 (-111%)  los=0.0%  ci=32.3  draw=32.3%  games=396/1000  tasks=24/24

Net: `enyo-h16w768-v1-rc1.nn`

sprt-enyo-h16w768-v1-rc1.nn-vs-reference.net-1000-20260720-165443 20 Jul 16:54 · 8m
"candidate=enyo-h16w768-v1-rc1.nn vs reference=reference.net"
elo=-154.1  llr=-2.32/2.20 (-105%)  los=0.0%  ci=38.7  draw=32.0%  games=312/1000  tasks=24/24

## F: SF-like 32x11-1024-o8, factorised (`sf32`)

Net: `enyo-sf32-v1-rc2.nn`

sprt-enyo-sf32-v1-rc2.nn-vs-reference.net-1000-20260720-122638 20 Jul 12:26 · 12m
"candidate=enyo-sf32-v1-rc2.nn vs reference=reference.net"
elo=-87.7  llr=-2.25/2.20 (-102%)  los=0.0%  ci=26.6  draw=34.8%  games=514/1000  tasks=24/24

Net: `enyo-sf32-v1-rc1.nn`

sprt-enyo-sf32-v1-rc1.nn-vs-reference.net-1000-20260720-072332 20 Jul 07:23 · 9m
"candidate=enyo-sf32-v1-rc1.nn vs reference=reference.net"
elo=-134.5  llr=-2.39/2.20 (-109%)  los=0.0%  ci=34.0  draw=32.5%  games=366/1000  tasks=24/24

## D: 16x12-1024-o8, factorised (`h16`)

Net: `enyo-h16-v1-rc2.nn`

sprt-enyo-h16-v1-rc2.nn-vs-reference.net-1000-20260721-093650 21 Jul 09:36 · 20m
"candidate=enyo-h16-v1-rc2.nn vs reference=reference.net"
elo=-48.9  llr=-2.27/2.20 (-103%)  los=0.0%  ci=21.9  draw=12.9%  games=922/1000  tasks=24/24

Direct seed diagnostic (the supplied Forge state still said `active` although
all 1,000 games and 24 tasks were complete):

sprt-enyo-h16-v1-rc2.nn-vs-enyo-h16-v1-rc1.nn-1000-20260721-102937 21 Jul 10:29 · 24m
"candidate=enyo-h16-v1-rc2.nn vs reference=enyo-h16-v1-rc1.nn"
elo=+4.9  llr=0.15/2.20 (7%)  los=70.3%  ci=17.9  draw=34.0%  games=1000/1000  tasks=24/24

Net: `enyo-h16-v1-rc1.nn`

sprt-enyo-h16-v1-rc1.nn-vs-reference.net-1000-20260719-135931 19 Jul 13:59 · 8m
"candidate=enyo-h16-v1-rc1.nn vs reference=reference.net"
elo=-142.6  llr=-2.31/2.20 (-105%)  los=0.0%  ci=35.9  draw=36.4%  games=324/1000  tasks=24/24

## C: 8x12-1024-o8, factorised (`h8`)

Net: `enyo-h8-v1-rc2.nn`

sprt-enyo-h8-v1-rc2.nn-vs-reference.net-1000-20260719-090924 19 Jul 09:09 · 10m
"candidate=enyo-h8-v1-rc2.nn vs reference=reference.net"
elo=-116.9  llr=-2.27/2.20 (-103%)  los=0.0%  ci=34.6  draw=27.1%  games=410/1000  tasks=24/24

Net: `enyo-h8-v1-rc1.nn`

sprt-enyo-h8-v1-rc1.nn-vs-reference.net-1000-20260719-083358 19 Jul 08:33 · 11m
"candidate=enyo-h8-v1-rc1.nn vs reference=reference.net"
elo=-107.0  llr=-2.44/2.20 (-111%)  los=0.0%  ci=28.5  draw=36.2%  games=442/1000  tasks=24/24

## B: 4x12-1024-o8, factorised (`h4`)

Net: `enyo-h4-v1-rc2.nn`

sprt-enyo-h4-v1-rc2.nn-vs-reference.net-1000-20260719-000119 19 Jul 00:01 · 10m
"candidate=enyo-h4-v1-rc2.nn vs reference=reference.net"
elo=-110.8  llr=-2.28/2.20 (-104%)  los=0.0%  ci=31.5  draw=31.3%  games=418/1000  tasks=24/24

Net: `enyo-h4-v1-rc1.nn`

sprt-enyo-h4-v1-rc1.nn-vs-reference.net-1000-20260718-213427 18 Jul 21:34 · 10m
"candidate=enyo-h4-v1-rc1.nn vs reference=reference.net"
elo=-115.6  llr=-2.31/2.20 (-105%)  los=0.0%  ci=31.6  draw=32.1%  games=402/1000  tasks=24/24

## A: 1x12-1024-o8, unfactorised (`h1`)

Net: `enyo-h1-v1-rc2.nn`

sprt-enyo-h1-v1-rc2.nn-vs-reference.net-1000-20260718-183535 18 Jul 18:35 · 7m
"candidate=enyo-h1-v1-rc2.nn vs reference=reference.net"
elo=-190.8  llr=-2.32/2.20 (-105%)  los=0.0%  ci=50.1  draw=28.8%  games=264/1000  tasks=24/24

Net: `enyo-h1-v1-rc1.nn`

sprt-enyo-h1-v1-rc1.nn-vs-reference.net-1000-20260718-174636 18 Jul 17:46 · 8m
"candidate=enyo-h1-v1-rc1.nn vs reference=reference.net"
elo=-146.9  llr=-2.27/2.20 (-103%)  los=0.0%  ci=38.8  draw=31.8%  games=318/1000  tasks=24/24
