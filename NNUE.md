# Enyo NNUE

```sh
• Current native Enyo NNUE:

  features:      12,288
  hidden width:  1,024
  L1 input:      2 * 1024 = 2,048
  L2:            16
  L3/output:     scalar

  So the big accumulator matrix is:

  12,288 features * 1,024 hidden = 12,582,912 int16 weights

  Plus:

  1,024 input biases
  2,048 * 16 int8 l1 weights
  16 l1 biases
  16 l2 weights
  1 l2 bias
  1 output weight
  1 output bias

  current .nn size: 25,203,012 bytes.
```

`README.md` describes the current candidate creation workflow and command-line
entry points.

This file explains what those steps mean for the NNUE itself: weights, hidden
neurons, king buckets, packed tensors, accumulator updates, export, validation,
and SPRT.


Process map:

```text
Step 1  self-play
Step 2  row extraction and filtering
Step 3  signed bucket sampling
Step 4  teacher labeling
Step 5  tensor packing
Step 6  training and .nn export
Step 7  static validation
Step 8  replay gates
Step 9  SPRT
```

So this line:

```text
accumulator[perspective][0..1023] += input_weights[feature][0..1023]
```

means:

## How Training Works

Training is supervised learning: for many FENs, make the NNUE prediction closer
to a teacher score.

The short version:

```text
FEN
  -> active features
  -> accumulator
  -> dense head
  -> predicted centipawn score
  -> compare with teacher score
  -> adjust weights
```

### 1. Start With A Labeled Position

Each training row gives a position and a target score:

```text
FEN position
teacher score
optional WDL/result context
source metadata
```

One row after Step 4 teacher labeling looks like:

```json
{
  "fen": "8/p7/3n1p1p/P1k5/2p2KPP/8/8/1B6 w - - 9 55",
  "score": -288,
  "source_score": -259,
  "teacher": "stockfish",
  "teacher_depth": 16
}
```

`score` is the target being learned. `source_score` is the old score kept for
audit. The played PGN move is not the target.

### 2. Convert The FEN To Active Features

For one FEN, Enyo walks every occupied square. Each piece creates one active
feature per perspective.

Example with 30 pieces on the board:

```text
white perspective: 30 active features
black perspective: 30 active features
```

Each active feature is an input-weight row address:

```text
feature = perspective + king bucket + piece/color type + piece square
```

### 3. Build The Accumulator

The accumulator is the first hidden layer after active feature rows have been
summed.

For one perspective:

```text
accumulator = input_biases
for active_feature in position:
    accumulator += input_weights[active_feature]
```

Expanded for one hidden slot:

```text
acc[i] = input_bias[i]
       + input_weights[feature_a][i]
       + input_weights[feature_b][i]
       + input_weights[feature_c][i]
       + ...
```

So this line:

```text
accumulator[perspective][0..1023] += input_weights[feature][0..1023]
```

means:

```text
for every hidden slot i from 0 to 1023:
    add the learned contribution for this active feature to accumulator[i]
```

### 4. Run The Dense Head

The engine combines the side-to-move accumulator and opponent accumulator:

```text
us[1024] + them[1024] -> 2048 dense-head inputs
```

Then the dense head computes:

```text
2048 values -> L1 16 values -> L2 32 values -> 1 centipawn score
```

That final score is the NNUE prediction for the FEN.

### 5. Compare Prediction With Teacher

Example:

```text
teacher target     = +80 cp
current prediction = +20 cp
error              = -60 cp
training direction = raise this position's predicted score
```

The loss function turns that error into a number PyTorch can minimize. Huber
and MPE are different ways to measure that error.

### 6. Adjust Weights

Backpropagation computes which weights contributed to the error. The optimizer
nudges them slightly.

For the input layer, only active feature rows get direct updates for that row:

```text
input_weights[active_feature] changes slightly
```

The dense-head weights also change:

```text
L1 weights
L1 biases
L2 weights
L2 biases
output weights
output bias
```

Training repeats this for many batches and epochs.

### 7. Export The Trained Weights

After training, the learned numbers are written to a `.nn` file. Enyo loads
that file and treats those weights as fixed during search.

Important distinction:

```text
training:
    changes weights in model.nn

search:
    keeps weights fixed
    changes accumulator values by adding/subtracting fixed weight rows
```

### 8. How Search Uses The Same Accumulator

During a game, recomputing the accumulator from scratch at every search node
would be too slow. NNUE is "efficiently updatable" because most moves only
change a few active features.

Quiet move:

```text
subtract old feature row for moved piece
add new feature row for moved piece
```

Capture:

```text
subtract old feature row for moved piece
subtract captured piece feature row
add new feature row for moved piece
```

King bucket change:

```text
refresh affected perspective, because many feature addresses changed
```

Simplified engine flow:

```text
search_node(board):
    if leaf:
        return evaluate(board)

    for move in legal_moves:
        apply_move(board, move)
        score = -search_node(board)
        undo_move(board, move)
```

Simplified NNUE flow:

```text
apply_move:
    store lazy accumulator delta for the child ply

evaluate:
    materialize lazy accumulator updates if needed
    choose us/them accumulators from side-to-move
    run dense head
    return centipawn score
```

## Step 1-4 Summary

These are performed by `build.py create`; this is only the short meaning.

Step 1, self-play:

```text
Generate Enyo-vs-Enyo games from the current reference engine.
Output: selfplay.pgn
```

Step 2, row extraction and filtering:

```text
Convert PGN into JSONL rows.
Remove unusable rows: missing score, mate score, timeout/emergency, duplicates.
Output: selfplay.jsonl
```

Step 3, signed bucket sampling:

```text
Sample balanced positive and negative score buckets.
Avoid training mostly on near-zero positions.
Output: source_signed.jsonl
```

Step 4, teacher labeling:

```text
Ask Stockfish for stronger scores, usually depth 16 or 18.
Output: labeled.jsonl or shards/label.*.jsonl
```

## Step 5: Tensor Packing

JSONL is readable but slow. Step 5 converts FEN rows into numeric arrays that
PyTorch can load efficiently.

Script:

```sh
tools/pack/pack.py build
```

Typical output:

```text
packed/
  white_features.npy
  black_features.npy
  counts.npy
  stm.npy
  score.npy
  wdl.npy
  phase_scale.npy
  source_id.npy
  meta.json
```

Meaning:

```text
white_features.npy  active feature indices from White perspective
black_features.npy  active feature indices from Black perspective
counts.npy          number of valid feature slots for each row
stm.npy             side to move
score.npy           teacher centipawn target
wdl.npy             optional WDL/result target
source_id.npy       lets validation split by data source
```

Packed tensors are not a trained net. They are the training rows in a faster
numeric form.

## Step 6: Training And Export

Script:

```sh
tools/train/train.py run
```

Step 6 loads packed tensors, initializes the model, trains weights, and exports:

```text
model.pt  PyTorch checkpoint
model.nn  Enyo runtime net
```

Common arguments:

```text
--init-from-nn     starting net to fine-tune from
--objective        loss/objective, usually huber or mpe25
--lr               learning rate, the update size
--target-clamp     clamp extreme teacher scores
--epochs           passes over the sampled training set
--batch-size       rows per optimizer step
```

Huber:

```text
Centipawn objective.
Small errors behave like squared error.
Large errors behave closer to absolute error.
Useful for clean d16/d18 teacher labels.
```

MPE/WDL-style:

```text
Probability-shaped objective.
Tries to care more about practical eval shape than exact huge centipawn values.
Useful for broader mixed sources, but must be proven by SPRT.
```

Target clamp:

```text
score +1800 with clamp 1000 becomes +1000
score -1700 with clamp 1000 becomes -1000
```

The goal is to stop already-won or already-lost positions from dominating the
training update.

Exported `.nn` is a binary runtime file. Enyo loads it with:

```text
setoption name nnue_file value /path/to/model.nn
```

## Step 7: Static Validation

Script:

```sh
tools/validate/validate.py static
```

Static validation compares the starting net and candidate net on held-out rows.
It can reject bad candidates, but it cannot prove Elo.

Useful metrics:

```text
MAE    average absolute centipawn error
MSE    squared error, sensitive to big misses
sign   how often net and target agree which side is better
bias   average prediction minus target
slope  eval calibration; low means compressed, high means exaggerated
corr   whether prediction ordering tracks target ordering
```

Good static result:

```text
candidate improves same-source MAE
candidate does not badly regress Lichess/binpack/self-play validation
sign drop is tiny
slope/bias are not distorted
```

Bad static result:

```text
MAE improves only on one source
sign drops materially
candidate compresses evals too much
candidate distorts one source to fit another
```

## Step 8: Replay Oracle Diff Gate

Replay gates are sanity checks on real Enyo games.

They answer:

```text
Does this candidate choose better or worse moves than the current reference
engine on known problem positions, as judged by an oracle engine?
```

They do not prove strength. They only catch bad behavior before spending time on
SPRT.

Command shape:

```sh
replay \
  --candidate <candidate-wrapper> \
  --reference <reference-wrapper> \
  --csv \
  ~/code/cpp/chess/enyo/bugs
```

For current Enyo builds, replay defaults are the intended NNUE gate settings:

```text
candidate budget = fixed 100000 nodes
oracle budget    = 200000 nodes
oracle engine    = stockfish
threads          = 1
```

Use `--fixed-movetime` only for old engine binaries that do not support
`go nodes`.

Scripts should aggregate the CSV `candidate_loss`, `reference_loss`, and `diff`
fields. The old logged move is only historical context; the useful gate is
candidate-vs-reference.

## Step 9: SPRT

SPRT is the final strength test.

Script:

```sh
tools/validate/validate.py sprt --net /path/to/model.nn --tag my_candidate
```

It runs:

```text
candidate = same Enyo binary + candidate nnue_file
reference = same Enyo binary + current reference nnue_file
```

This isolates the net change from engine-code changes.

Interpretation:

```text
negative early       stop and archive
0..+5 Elo            weak/inconclusive
+5..+8 Elo           rerun only if static metrics are clean
+8 Elo or SPRT H1    candidate worth confirmation
+3..+5 Elo           real only after very large or slower confirmation
```

Do not promote a net from MAE alone. Promote only after game strength is
confirmed.

## Workflow Entry Points

Use the root command for normal candidate creation:

```sh
./build.py create --help
```

It writes a `runs/<run-name>/config.json` pipeline and executes:

```text
posgen -> score -> pack -> train
```

Validation is explicit:

```sh
tools/validate/validate.py static --help
tools/validate/validate.py failure-suite --help
tools/validate/validate.py sprt --help
```

`tools/validate/run_net_sprt_pwa.sh` is kept as the low-level SPRT runner used
by `tools/validate/validate.py sprt`. The older one-off experiment launchers
were removed; they were historical recipes, not the current workflow.

Running the exact same training command repeatedly mostly tests randomness. A
meaningful new experiment changes at least one of:

```text
data source
teacher depth
bucket distribution
objective
learning rate
target clamp
epochs
starting net
```
