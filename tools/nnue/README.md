# core NNUE utilities

This directory contains low-level NNUE implementation utilities used by the
higher-level phase tools. It is not the normal entry point for creating a
candidate.

Use these commands instead:

```sh
./build.py create --help
tools/posgen/posgen.py --help
tools/score/score.py --help
tools/pack/pack.py --help
tools/train/train.py --help
tools/validate/validate.py --help
```

## Kept Here

Reusable core utilities:

```text
model.py                  PyTorch network definition
dataset.py                JSONL/packed dataset loading
pack_dataset.py           JSONL -> packed tensor arrays
train.py                  low-level training/export implementation
eval_dataset.py           static metric evaluation
export.py                 .nn export support
enyo_nnue.py              feature/index constants matching Enyo
roundtrip.py              loader/exporter sanity checks
label_with_uci.py         generic UCI teacher labeling
import_lichess_eval.py    Lichess eval DB import
binpack_to_jsonl.cpp      Stockfish binpack import
mix_jsonl.py              source mixing helper
sample_*.py               bucket/instability sampling helpers
replay_failure_suite.py   candidate/reference/oracle replay gate
run_net_sprt_pwa.sh       low-level SPRT runner used by validate.py
```

The old one-off `run_*_pwa.sh` experiment launchers were removed. Their useful
behavior should live in `build.py`, the phase tools, or run metadata under
`runs/`.

## Direct Examples

Roundtrip check:

```sh
python3 tools/nnue/roundtrip.py \
  ~/code/cpp/chess/enyo/nnue/berserk-d43206fe90e4.nn
```

Target/data quality check:

```sh
python3 tools/nnue/audit_targets.py \
  runs/imported/sf_d12_20m_20260510_115338/score/labeled.jsonl \
  --rows 100000
```

Search-instability sample:

```sh
python3 tools/nnue/sample_search_instability.py \
  --input runs/imported/<run>/score/labeled.jsonl \
  --output runs/<new-run>/assets/instability.jsonl \
  --engine ~/local/bin/stockfish \
  --low-depth 8 \
  --high-depth 16 \
  --min-delta-cp 80 \
  --bestmove-change \
  --max-output 500000
```

Low-level SPRT runner:

```sh
NET=/path/to/model.nn TAG=my_candidate tools/nnue/run_net_sprt_pwa.sh
```

Prefer `tools/validate/validate.py sprt` unless you need the low-level runner
directly.
