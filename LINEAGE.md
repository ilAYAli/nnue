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
| enyo-5.1.0-rc2 | pwa-5090 | enyo-5.0.0-rc1 | switch continuation corpus to self-play-1.2 shuffled | accepted | c4aa34c5998848e5784b730e97f3e7a0054a64f7f88eab7ef086a32e484707a5 | af9f1e23 | +64.2 / 15.9 | -263.1 / 9.3 | 4000 | Selected parent; beat rc3 by +13.2 / 6.8 at 3940 games, LLR 2.28/2.20, LOS 100.0%. |
| enyo-5.1.0-rc3 | pwa-llm | enyo-5.0.0-rc1 | raise WDL from 0.05 to 0.10 | accepted | e04a17b19a1201bab0b9c49c0c463a44566d75dc20b7971b6a05160110c1dcc9 | b6d228bf | +64.7 / 17.0 | -281.5 / 9.6 | 4000 | Passed but not selected; lost the direct tie-break to rc2. |
| enyo-5.2.0-rc1 | pwa-llm | enyo-5.1.0-rc2 | second full dose on the winning self-play-1.2 corpus | accepted | 6f393ea228c588934eb0c64c602409b5a6908a6ef6bd75b501deb65091cf9a8b | e6da210c | +38.2 / 12.6 | -254.2 / 9.0 | 4000 | Selected parent; beat rc2 by +6.8 / 5.8 over 5000 direct games. |
| enyo-5.2.0-rc2 | pwa-5090 | enyo-5.1.0-rc2 | raise WDL from 0.05 to 0.10 | accepted | bbf2d21f7f3eb0407224432fe0853bdab26b8155358582ca2872c57b543ea824 | d4403d9d | +36.1 / 13.6 | -252.6 / 8.9 | 4000 | Passed but not selected; lost the direct tie-break to rc1. |
| enyo-5.2.0-rc3 | pwa-llm | enyo-5.1.0-rc2 | raise WDL from 0.05 to 0.075 | accepted | 5349e54fc9267b386805c2a45ae4d6c092dffd8fb7ea57b4f1941a6ea4fb1a10 | 3e91620a | +30.7 / 11.0 | -259.3 / 9.2 | 4000 | Passed at the positive cap but not selected. |
| enyo-5.3.0-rc1 | pwa-llm | enyo-5.2.0-rc1 | third full dose with the selected WDL 0.05 regimen | accepted | bc0102cefb7a44313d2b59451670595836a885607c64f5c6d4ef177009f3fbc4 | c54db7e4 | +26.2 / 12.5 | -238.0 / 8.8 | 4000 | Passed but not selected on the completed fixed-SF point estimate. |
| enyo-5.3.0-rc2 | pwa-5090 | enyo-5.2.0-rc1 | raise final LR from 0.000005 to 0.000010 | accepted | 4ed3eca977c10cb2a32cfc086c45d686554d1e162158743088397dd3b5b2527e | 5b851c74 | +28.7 / 11.6 | -232.0 / 8.5 | 4000 | Selected on identical fixed-SF point estimate; direct confirmation is running. |
| enyo-5.4.0-rc1 | pwa-llm | enyo-5.3.0-rc2 | fourth full dose with the selected final LR 0.000010 regimen | accepted | 26be08a3ce654d855ed206a7c1c1423889d6536a544da9d6f27349736701bff2 | 947cc252 | +8.8 / 10.7 | -220.2 / 8.1 | 4000 | Selected on identical fixed-SF point estimate; direct confirmation is running. |
| enyo-5.4.0-rc2 | pwa-5090 | enyo-5.3.0-rc2 | raise final LR from 0.000010 to 0.000020 | accepted | 18180c97cdb5d7334e2ba2496171863736b86a2c52a23ea2b216e3a1a490af8a | 0d25473a | +9.7 / 10.5 | -224.8 / 8.4 | 4000 | Passed but not selected on the completed fixed-SF point estimate. |
| enyo-5.5.0-rc1 | pwa-llm | enyo-5.4.0-rc1 | fifth full dose with the selected final LR 0.000010 regimen | reserved | — | — | — | — | 0 | Same architecture, corpus, WDL, and optimizer; `continue_from` rc1. |
| enyo-5.5.0-rc2 | pwa-5090 | enyo-5.4.0-rc1 | lower initial LR from 0.0010 to 0.0005 | reserved | — | — | — | — | 0 | Same architecture, corpus, dose, WDL, and final LR; `continue_from` rc1. |
