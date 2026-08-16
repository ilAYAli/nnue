# Enyo 5 Lineage

`Parent SPRT` is against the immediately preceding selected net.
`SF` is the fixed 4,000-game benchmark against `nn-0ee0657fb25e.nnue`

Architecture 5: SCReLU dense activation (`relu-screlu-residual`).
Architecture 6: Architecture 5 plus the activated L2→output skip.
Architecture 7: Architecture 6 plus 32 input buckets.
Current selected parent: `enyo-7.1.0-rc2` (+5.7 ±5.4 vs `enyo-7.1.0-rc1`; −150.3 ±7.4 vs fixed SF, `enyo_f2a0417`).

|  # | Selected run    | Commit   | Host     | Parent SPRT  | SF            | Change
|----|-----------------|----------|----------|--------------|---------------|--------------------------------
|  0 | enyo-5.0.0-rc1  | 11aa8bc4 | pwa-llm  | —            | -315.2 ± 10.6 | scratch SCReLU root
|  1 | enyo-5.1.0-rc2  | 715837c6 | pwa-5090 | +64.2 ± 15.9 | -263.1 ± 9.3  | shuffled self-play-1.2 corpus
|  2 | enyo-5.2.0-rc1  | d2da4c89 | pwa-llm  | +38.2 ± 12.6 | -254.2 ± 9.0  | next full dose
|  3 | enyo-5.3.0-rc2  | b6db90d0 | pwa-5090 | +28.7 ± 11.6 | -232.0 ± 8.5  | final LR 0.000005 → 0.000010
|  4 | enyo-5.4.0-rc1  | f09a933b | pwa-llm  | +8.8 ± 10.7  | -220.2 ± 8.1  | next full dose
|  5 | enyo-5.5.0-rc1  | bf950072 | pwa-llm  | +11.1 ± 10.6 | -216.1 ± 8.4  | next full dose
|  6 | enyo-5.6.0-rc2  | 074dc6b9 | pwa-5090 | +24.8 ± 10.4 | -204.8 ± 8.1  | initial LR 0.0010 → 0.0020
|  7 | enyo-5.7.0-rc2  | 73b965c3 | pwa-5090 | +19.0 ± 10.1 | -191.2 ± 8.1  | initial LR 0.0020 → 0.0030
|  8 | enyo-5.8.0-rc1  | e0ab687d | pwa-llm  | +13.0        | -192.6 ± 8.1  | next full dose
|  9 | enyo-5.9.0-rc6  | 0ec2e1fc | pwa-5090 | +7.6 ± 10.2  | -183.9 ± 7.7  | two corpus passes
| 10 | enyo-5.10.0-rc2 | 568b0c3c | pwa-5090 | +9.0 ± 10.7  | -185.3 ± 8.0  | four corpus passes
| 11 | enyo-6.0.0-rc1  | 486c6e59 | pwa-llm  | +2.8 ± 10.3  | -165.1 ± 7.4  | activated L2-to-output skip
| 12 | enyo-6.1.0-rc2  | 2741cfd1 | pwa-llm  | +7.9 ± 10.1  | -170.5 ± 7.7  | two corpus passes
| 13 | enyo-6.2.0-rc4  | 18dd99b0 | pwa-5090 | +2.1 ± 10.6  | -164.3 ± 7.5  | initial LR 0.0030 → 0.0040
| 14 | enyo-6.3.0-rc3  | 1aa663b7 | pwa-llm  | +12.0 ± 10.4 | -148.6 ± 7.5  | final LR 0.000010 → 0.000005; tie-break +6.6 ± 5.5 vs rc4
| 15 | enyo-6.4.0-rc1  | cd7728bc | pwa-llm  | +0.7 ± 10.2  | -156.1 ± 7.4  | three corpus passes; tie-break +4.4 ± 5.6 vs rc2
| 16 | enyo-7.0.0-rc2  | dde4b621 | pwa-llm  | +9.3          | -158.9 ± 7.4  | 32 input buckets; explicit initialization
| 17 | enyo-7.1.0-rc1  | fc024567 | pwa-llm  | +2.3 ± 9.9   | -147.2 ± 7.3  | full continuation
| 18 | enyo-7.2.0-rc1  | 296f72bd | pwa-llm  | +4.2          | -157.0 ± 7.6  | second full continuation
| 19 | enyo-7.1.0-rc2  | 22a4a975 | pwa-5090 | +5.7 ± 5.4    | -150.3 ± 7.4  | independent continuation trajectory

Reserved: none.

# Enyo 1 Lineage (recovered)

`Parent SPRT` is historical and is not comparable to the Enyo-5/6 results.
Rows are the selected direct-weight path; omitted versions are rejected or
non-ancestor sibling experiments. Scale rows are deterministic export
transforms, not training runs.

|  # | Selected run                      | Commit   | Host | Parent SPRT   | SF | Change
|----|-----------------------------------|----------|------|---------------|----|--------------------------------
|  0 | enyo-scratch-broad-1.0.0-rc1      | 813e229b | —    | —             | —  | random Pylon root; 2.8B positions; 65,536 SB; LR 0.0010 → 0.000005
|  1 | enyo-scratch-long-1.0.0-rc1       | a0dc31b6 | —    | +93.2         | —  | random weight root; broad Bullet corpus; 196,608 SB
|  2 | enyo-scratch-long-1.1.0-rc1       | 7c7a0764 | —    | +45.9         | —  | 196,608 SB
|  3 | enyo-scratch-long-1.2.0-rc1       | a1c8368a | —    | +25.8         | —  | 196,608 SB
|  4 | enyo-scratch-long-1.3.0-rc1       | 1c22805e | —    | +16.0         | —  | 196,608 SB
|  5 | enyo-scratch-long-1.4.0-rc1       | d6c42e4c | —    | +7.4          | —  | 196,608 SB
|  6 | enyo-scratch-long-1.5.0-rc1       | 8cea17a7 | —    | +0.5          | —  | 98,304 SB
|  7 | enyo-1.0.0-rc1                    | 9d829ac0 | —    | +4.9          | —  | 98,304 SB
|  8 | enyo-1.1.0-rc2                    | f054980b | —    | +9.5          | —  | WDL 0.05 → 0.15; broad Bullet corpus; 98,304 SB
|  9 | enyo-1.2.0-rc1                    | 9efb7de5 | —    | +3.0          | —  | broad Bullet corpus; 98,304 SB
| 10 | enyo-1.3.0-rc2                    | 5572685c | —    | +13.4         | —  | broad Bullet corpus; 98,304 SB
| 11 | enyo-1.4.0-rc2                    | 15cc8afb | —    | +2.1          | —  | broad Bullet corpus; 98,304 SB
| 12 | enyo-1.5.0-rc2                    | 11a87ee5 | —    | +11.8         | —  | broad Bullet corpus; 32,768 SB
| 13 | enyo-1.6.0-rc2                    | 0e829bfc | —    | +13.9         | —  | broad Bullet corpus; 32,768 SB
| 14 | enyo-1.7.0-rc3                    | db3687ff | —    | +5.3          | —  | LR 0.00025; broad Bullet corpus; 32,768 SB
| 15 | enyo-1.11.0-rc2                   | 97757ff5 | —    | +7.7          | —  | LR 0.000125; broad Bullet corpus; 16,384 SB
| 16 | enyo-1.12.0-rc1                   | dd74d02b | —    | +6.7          | —  | broad Bullet corpus; 16,384 SB
| 17 | enyo-1.13.0-rc2                   | 236ed823 | —    | +7.9          | —  | FarseerT76; WDL 0.30; 256 SB
| 18 | enyo-1.14.0-rc2                   | 4eef094b | —    | +2.1          | —  | FarseerT76; WDL 0.05; 256 SB
| 19 | enyo-1.15.0-rc2                   | d09170e7 | —    | +12.3         | —  | FarseerT76; LR 0.0001; 256 SB
| 20 | enyo-1.16.0-rc3                   | 79cd05cb | —    | +4.9          | —  | T60T70wIsRightFarseer; 256 SB
| 21 | enyo-1.20.0-rc12                  | 586a5c29 | —    | +2.1          | —  | Pylon; output-only; 512 SB
| 22 | enyo-1.28.0-rc16                  | f37067d2 | —    | +10.2         | —  | Pylon; input-only; LR 0.00001; 256 SB
| 23 | enyo-1.30.0-rc3                   | —        | —    | +25.3 ± 13.9  | —  | output-head x0.48; grid: rc1 .52, rc2 .56, rc4 .50, rc5 .46
| 24 | enyo-1.30.0-rc3-unscaled          | —        | —    | —             | —  | inverse scale; input/L1/L2 identical across grid
| 25 | enyo-1.31.0-rc57                  | 3c464a3b | —    | +8.6          | —  | Stockfish-static relabel of 46.4M self-play positions; 354 SB
| 26 | enyo-1.32.0-rc10                  | 650a4fcc | —    | +3.3          | —  | recalibrated nodes5000pv2 labels; 4,000 SB
