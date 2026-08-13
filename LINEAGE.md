# Enyo 5 Lineage

`Parent SPRT` is against the immediately preceding selected net.
`SF` is the fixed 4,000-game benchmark against `nn-0ee0657fb25e.nnue`

| # | Selected run   | Commit   | Host     | Parent SPRT  | SF            | Change
|---|----------------|----------|----------|--------------|---------------|--------------------------------
| 0 | enyo-5.0.0-rc1 | 11aa8bc4 | pwa-llm  | —            | -315.2 ± 10.6 | scratch SCReLU root
| 1 | enyo-5.1.0-rc2 | 715837c6 | pwa-5090 | +64.2 ± 15.9 | -263.1 ± 9.3  | shuffled self-play-1.2 corpus
| 2 | enyo-5.2.0-rc1 | d2da4c89 | pwa-llm  | +38.2 ± 12.6 | -254.2 ± 9.0  | next full dose
| 3 | enyo-5.3.0-rc2 | b6db90d0 | pwa-5090 | +28.7 ± 11.6 | -232.0 ± 8.5  | final LR 0.000005 → 0.000010
| 4 | enyo-5.4.0-rc1 | f09a933b | pwa-llm  | +8.8 ± 10.7  | -220.2 ± 8.1  | next full dose
| 5 | enyo-5.5.0-rc1 | bf950072 | pwa-llm  | +11.1 ± 10.6 | -216.1 ± 8.4  | next full dose
| 6 | enyo-5.6.0-rc2 | 074dc6b9 | pwa-5090 | +24.8 ± 10.4 | -204.8 ± 8.1  | initial LR 0.0010 → 0.0020
| 7 | enyo-5.7.0-rc2 | 73b965c3 | pwa-5090 | +19.0 ± 10.1 | -191.2 ± 8.1  | initial LR 0.0020 → 0.0030

## Current selection and reservations

`enyo-5.9.0-rc6` is selected: it passed its parent SPRT by +7.6 ± 10.2 Elo,
benchmarked at -183.9 ± 7.7 versus SF, and beat one-pass rc5 by +5.7 ± 5.6
Elo over 5,000 games.

| Run | Host | Parent | Single variable |
|---|---|---|---|
| enyo-5.9.0-rc1–rc2 | — | — | void: invalid parent/configuration pairing |
| enyo-5.9.0-rc3–rc4 | — | — | void: unsupported corpus ablations |
| enyo-5.9.0-rc5 | pwa-llm | enyo-5.8.0-rc1 | rejected: one corpus pass (25,702 superbatches) |
| enyo-5.9.0-rc6 | pwa-5090 | enyo-5.8.0-rc1 | selected: two corpus passes (51,404 superbatches) |
| enyo-5.10.0-rc1 | pwa-llm | enyo-5.9.0-rc6 | dose: three corpus passes (77,106 superbatches) |
| enyo-5.10.0-rc2 | pwa-5090 | enyo-5.9.0-rc6 | dose: four corpus passes (102,808 superbatches) |
