# Enyo NNUE Trainer

Run commands from the `nnue` repo root unless noted otherwise.

This trains or fine-tunes Enyo's Berserk-format `1024` hidden-neuron
network and exports a `.nn` file loadable through:

```text
setoption name nnue_file value nnue/<file>.nn
```

Typical first run is fine-tuning the current Berserk net on Enyo
self-play rows:

```bash
python3 tools/nnue/train.py \
  --data ~/tmp/enyo_selfplay/50k_d8_sharded_20260506_113050/selfplay_d8_50k.jsonl \
  --init-from-nn ~/code/cpp/chess/enyo/nnue/berserk-d43206fe90e4.nn \
  --max-rows 1000000 \
  --val-rows 50000 \
  --device cuda \
  --epochs 20 \
  --batch-size 4096 \
  --lr 1e-5 \
  --objective mpe25 \
  --wdl-lambda 0.75 \
  --out ~/tmp/nnue_selfplay.pt \
  --out-nn ~/tmp/nnue_selfplay.nn
```

Roundtrip check for loader/exporter correctness:

```bash
python3 tools/nnue/roundtrip.py \
  ~/code/cpp/chess/enyo/nnue/berserk-d43206fe90e4.nn
```

Target/data quality check before training:

```bash
python3 tools/nnue/audit_targets.py \
  ~/tmp/enyo_teacher/sf_d12_20m_20260510_115338/labeled.jsonl \
  --rows 100000
```

Import filtered Lichess eval DB rows:

```bash
python3 tools/nnue/import_lichess_eval.py \
  --input ~/data/lichess/lichess_db_eval.jsonl.zst \
  --output ~/tmp/enyo_teacher/lichess_eval_d18_standard/lichess_eval.jsonl \
  --rows 5000000 \
  --min-depth 18 \
  --max-abs-cp 1600 \
  --unique-fen
```

The importer converts Lichess eval DB centipawns from white POV to Enyo's
side-to-move training convention, and rejects non-standard material positions.
It can also signed-bucket sample while streaming the compressed file, avoiding a
large intermediate filtered JSONL:

```bash
python3 tools/nnue/import_lichess_eval.py \
  --input ~/code/cpp/chess/assets/lichess_db_eval.jsonl.zst \
  --output ~/tmp/enyo_teacher/lichess_eval_slice/lichess_eval_signed.jsonl \
  --min-depth 18 \
  --max-abs-cp 1600 \
  --bucket any0:any:0:25:150000 \
  --bucket pos25_75:pos:25:75:100000 \
  --bucket neg25_75:neg:25:75:100000
```

On pwa-5090, the standard external-data prep command is:

```bash
tools/nnue/run_lichess_eval_slice_pwa.sh
```

Import official Stockfish `.binpack` rows:

```bash
c++ -O3 -std=c++23 \
  -I ~/tmp/nnue-pytorch/data_loader/cpp/lib \
  tools/nnue/binpack_to_jsonl.cpp \
  -lfmt \
  -o ~/tmp/binpack_to_jsonl

~/tmp/binpack_to_jsonl \
  --input ~/code/cpp/chess/assets/test79-may2022-16tb7p-filter-v6-dd.min-mar2023.unmin.high-simple-eval-1k.min-v2.binpack \
  --output ~/tmp/enyo_teacher/binpack_test79/binpack.jsonl \
  --limit 5000000 \
  --max-abs-cp 1600
```

The converter emits Enyo JSONL rows, converts Stockfish's internal eval unit to
centipawns with `100 * score / 208`, and rejects illegal/extreme positions.

Mix Enyo self-play labels with imported Lichess eval rows:

```bash
python3 tools/nnue/mix_jsonl.py \
  --output ~/tmp/enyo_teacher/mixed_20m_selfplay_5m_lichess.jsonl \
  --seed 20260511 \
  --source ~/tmp/enyo_teacher/sf_d12_20m_20260510_115338/labeled.jsonl:20000000 \
  --source ~/tmp/enyo_teacher/lichess_eval_d18_standard/lichess_eval.jsonl:5000000
```

The mixer streams selected rows and intentionally does not do cross-source
FEN dedupe. Dedupe each input before mixing if exact uniqueness is required.

Sample search-instability rows:

```bash
python3 tools/nnue/sample_search_instability.py \
  --input ~/tmp/enyo_teacher/fresh_d16_labels_20260515_100955/selfplay.jsonl \
  --output ~/tmp/enyo_teacher/search_instability/unstable.jsonl \
  --engine ~/local/bin/stockfish \
  --low-depth 8 \
  --high-depth 16 \
  --min-delta-cp 80 \
  --bestmove-change \
  --max-output 500000
```

This selects positions where the teacher changes score or best move between
two depths, then writes rows labeled with the high-depth score. Use it for the
next data cycle when cp buckets alone stop producing Elo.

Notes:

- The model includes Enyo's `ScaleEval` phase scaling during training,
  because search uses the scaled value.
- `--init-from-nn` is the sane default. Training this architecture from
  scratch needs much more data and time.
- The current dataset loader is intentionally simple and loads selected
  JSONL rows into memory. Use `--max-rows` for JSONL pilots.
- For multi-million-row runs, first pack the JSONL to mmap arrays:

```bash
python3 tools/nnue/pack_dataset.py \
  --input ~/tmp/enyo_selfplay/d8_6m_20260509_165354/selfplay.jsonl \
  --out-dir ~/tmp/enyo_selfplay/d8_6m_20260509_165354/selfplay_packed
```

Then pass the packed directory to `--data`.

Hard move-choice gate:

```bash
python3 tools/nnue/build_move_gate.py \
  ~/code/cpp/chess/enyo/bugs \
  --output ~/tmp/enyo_hard_gate_cases.jsonl \
  --min-loss 70 \
  --max-loss 999

python3 tools/nnue/eval_move_gate.py \
  --cases ~/tmp/enyo_hard_gate_cases.jsonl \
  --engine ~/code/cpp/chess/assets/engines/reference \
  --baseline-net ~/code/cpp/chess/enyo/nnue/berserk-d43206fe90e4.nn \
  --candidate-net ~/tmp/enyo_teacher/<run>/<candidate>/model.nn
```

This is a fast pre-SPRT check for known Enyo failures. For each replay issue
with a FEN, it evaluates the child position after the played move and after the
teacher-best move. A useful candidate should increase the margin in favor of
the teacher-best move compared with the baseline net. This does not replace
SPRT, but it stops obviously wrong objectives before they burn match time.

Hard-case augmented training on `pwa-5090`:

```bash
tmux new-session -d -s nnue_test \
  'cd ~/code/cpp/chess/nnue; tools/nnue/run_hardcase_aug_pwa.sh'
```

That runner builds the hard gate cases, expands them into played/best child
positions, labels those child positions with Stockfish, repeats them into the
training mix, trains two low-LR Huber candidates, runs the hard move-choice
gate, and only starts SPRT if a candidate improves the hard gate.

Pairwise hard-case fine-tuning:

```bash
python3 tools/nnue/train_pairwise.py \
  --data ~/tmp/enyo_teacher/hardcase_aug_<run>/hardmajor_d16500000_bin350000_hard150000/packed \
  --pairs ~/tmp/enyo_teacher/hardcase_aug_<run>/hard_child_labeled.jsonl \
  --init-from-nn ~/code/cpp/chess/enyo/nnue/berserk-d43206fe90e4.nn \
  --epochs 8 \
  --lr 1e-6 \
  --pair-weight 1.0 \
  --device cuda \
  --workers 2 \
  --out ~/tmp/pairwise.pt \
  --out-nn ~/tmp/pairwise.nn
```

This trains the usual centipawn fit and adds a direct pairwise margin loss:
after the played move and after the teacher-best move, the net should score the
teacher-best child higher from the original mover's point of view. Use this
when broad MAE improves but the hard move-choice gate barely moves.
