# Enyo 5 Lineage

This is the canonical ancestry, reservation, and result ledger for the clean
`enyo-5` lineage. Do not reconstruct entries from memory. Record each reservation
on `pwa-llm` before launch and complete it from preserved artifacts and results.

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

## Experiments

| Run | Host | Parent | Variable | State | Net SHA-256 | Commit | Parent Elo / CI | SF Elo / CI | Games | Notes |
|---|---|---|---|---|---|---|---|---|---:|---|
| enyo-5.0.0-rc1 | pwa-llm | scratch | SCReLU v4 root; WDL 0.05; 221977 superbatches; preserved interleaved corpus | accepted | 1adb72ef065aa9ebe5cc3c431b0c6301a8b7b3605bec9bc1a90ef829cdd841ae | 8364ea6f | — | -315.2 / 10.6 | 4000 | Fixed-SF run `sprt-enyo_203ab1f-4000-20260809-175759`; no origin fields; random initialization. |
| enyo-5.1.0-rc1 | pwa-llm | enyo-5.0.0-rc1 | second full dose on the same interleaved corpus | accepted | 045e4b6c6de7360008d668dba43cdf4e16ef7cf9ce1a2dc992a292bdaa1116f0 | 5404c5d0 | +59.1 / 16.7 | -287.9 / 9.7 | 4000 | Parent SPRT passed at 688 games, LLR 2.28/2.20, LOS 100.0%; fixed-SF run `sprt-enyo_203ab1f-4000-20260810-002529`; selection pending rc2. |
| enyo-5.1.0-rc2 | pwa-5090 | enyo-5.0.0-rc1 | switch continuation corpus to self-play-1.2 shuffled | training | — | — | — | — | 0 | Same architecture, dose, WDL, and optimizer; `continue_from` root. Forge testing owned by pwa-llm. |
| enyo-5.1.0-rc3 | pwa-llm | enyo-5.0.0-rc1 | raise WDL from 0.05 to 0.10 | training | — | — | — | — | 0 | Same architecture, interleaved corpus, dose, and optimizer; `continue_from` root. Promotion waits for rc2 and rc3. |
