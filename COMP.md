# Architecture competition results

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

sprt-enyo-h16-v1-rc2.nn-vs-reference.net-1000-20260719-193607 19 Jul 19:36 · 9m
"candidate=enyo-h16-v1-rc2.nn vs reference=reference.net"
elo=-136.1  llr=-2.38/2.20 (-108%)  los=0.0%  ci=34.7  draw=33.9%  games=354/1000  tasks=24/24

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
