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
| enyo-5.0.0-rc1 | pwa-llm | scratch | SCReLU v4 root; WDL 0.05; 221977 superbatches; preserved interleaved corpus | training | — | — | — | — | 0 | No `continue_from` or `initialize_from`; random initialization. |
