# Enyo 5 Lineage

This is the canonical selected ancestry for the clean `enyo-5` lineage.
Experiment details, including rejected and void runs, remain in Git and preserved
run artifacts. Do not reconstruct entries from memory.

Fixed external target: `nn-0ee0657fb25e.nnue`.

## Lineage rules

- `enyo-5.0.0-rc1` is the sole scratch root and has no origin fields.
- Every descendant uses `continue_from` the selected proven parent.
- Siblings share one parent, have globally unique RCs, and change one meaningful
  variable each.
- Valid states are `reserved`, `training`, `accepted`, `rejected`, and `void`.
- Never reuse a run name. A `void` run has no invented Elo.
- Wait for all in-flight siblings before selecting the next parent.
- A `pwa-5090` winner becomes a parent only after its commit, net, provenance,
  and hashes are verified on `pwa-llm`.

## Selected ancestry

`Parent SPRT` is against the immediately preceding selected net. `SF` is the
fixed 4,000-game benchmark, shown as Elo ± CI. Commit links the exact config.

| Promotion | Selected run | Host | Change | Parent SPRT | SF | Commit |
|---:|---|---|---|---|---|---|
| 0 | enyo-5.0.0-rc1 | pwa-llm | scratch SCReLU root | — | -315.2 ± 10.6 | 11aa8bc4 |
| 1 | enyo-5.1.0-rc2 | pwa-5090 | shuffled self-play-1.2 corpus | +64.2 ± 15.9 | -263.1 ± 9.3 | 715837c6 |
| 2 | enyo-5.2.0-rc1 | pwa-llm | next full dose | +38.2 ± 12.6 | -254.2 ± 9.0 | d2da4c89 |
| 3 | enyo-5.3.0-rc2 | pwa-5090 | final LR 0.000005 → 0.000010 | +28.7 ± 11.6 | -232.0 ± 8.5 | b6db90d0 |
| 4 | enyo-5.4.0-rc1 | pwa-llm | next full dose | +8.8 ± 10.7 | -220.2 ± 8.1 | f09a933b |
| 5 | enyo-5.5.0-rc1 | pwa-llm | next full dose | +11.1 ± 10.6 | -216.1 ± 8.4 | bf950072 |
| 6 | enyo-5.6.0-rc2 | pwa-5090 | initial LR 0.0010 → 0.0020 | +24.8 ± 10.4 | -204.8 ± 8.1 | 074dc6b9 |
| 7 | enyo-5.7.0-rc2 | pwa-5090 | initial LR 0.0020 → 0.0030 | +19.0 ± 10.1 | -191.2 ± 8.1 | 73b965c3 |

## Pending selection

| Candidate | Host | Change | Parent SPRT | SF | Commit |
|---|---|---|---|---|---|
| enyo-5.8.0-rc1 | pwa-llm | next full dose | +13.0 | -192.6 ± 8.1 | e0ab687d |
| enyo-5.8.0-rc2 | pwa-5090 | initial LR 0.0030 → 0.0040 | +10.0 ± 10.2 | -191.1 ± 7.8 | c6056975 |
