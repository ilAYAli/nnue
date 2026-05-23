# Enyo NNUE

`README.md` describes the current candidate creation workflow and command-line
entry points.

This file explains what those steps mean for the NNUE itself: native Enyo
architecture, weights, hidden neurons, king buckets, packed tensors,
accumulator updates, export, validation, and SPRT.

## Current Native Enyo Design

This is the normal Enyo `.nn` runtime layout, not the experimental Bullet
runtime and not a Reckless copy.

Source of truth in Enyo is currently `src/nnue_model.hpp`.

```text
king buckets                 = 32
legacy king buckets          = 16, accepted by loader and expanded
piece/color types            = 12
squares                      = 64
input feature rows           = 32 * 12 * 64 = 24576
accumulator width            = 1024 per perspective
perspectives                 = 2
dense-head input             = 2048
dense hidden layer 1         = 16
dense hidden layer 2         = 32
output                       = 1 centipawn score
optional output buckets      = 8, only for bucketed-head files
optional threat branch       = compile-time experiment, disabled by default
```

Data path:

```text
active sparse features
  -> two 1024-wide perspective accumulators
  -> side-to-move accumulator + opponent accumulator = 2048 values
  -> 2048 -> 16
  -> 16 -> 32
  -> 32 -> 1
  -> output / 32
  -> phase scale and search-side eval handling in Enyo
```

Stored weights in the current 32-king-bucket native file:

```text
input weights        = 24576 * 1024 = 25165824 int16 values
input biases         = 1024 int16 values
L1 weights           = 2048 * 16 = 32768 int8 values
L1 biases            = 16 int32 values
L2 weights           = 16 * 32 = 512 float values
L2 biases            = 32 float values
output weights       = 32 float values
output bias          = 1 float value
total trained values = 25200209
.nn payload size     = 50368836 bytes, about 48.0 MiB
```

The current native lane is Enyo-owned: it is intended to train/export Enyo's
own `.nn` format. Bullet may be used as a faster trainer, but `--bullet-mode
enyo` must export a normal Enyo `model.nn` for this lane.

## Vocabulary

The concrete values below are the current Enyo NNUE architecture values from
the Enyo source tree, currently `src/nnue_model.hpp`.

`NNUE`

Efficiently updatable neural network. It is a neural evaluator whose first
layer can be updated by adding and subtracting feature weight rows after a move.

`feature`

One possible board fact the NNUE knows how to address: perspective, king
bucket, piece/color type, and piece square. A feature is an address into the
input-weight table.

Actual Enyo NNUE values:

```text
king buckets                 = 32
piece/color types            = 12
squares                      = 64
input feature rows           = 32 * 12 * 64 = 24576
active feature rows per view = up to one row per piece on the board
```

`active feature`

One input feature that is actually present in a specific FEN. The full network
has 24576 possible feature rows, but a normal position activates only the rows
for pieces currently on the board.

Actual Enyo behavior:

```text
piece list                  = every occupied square, including kings
active rows per perspective = number of pieces on the board
active rows for both views  = 2 * number of pieces on the board
start position              = 32 active rows per view, 64 total
king-and-pawn endgame       = fewer active rows, because fewer pieces exist
```

`king bucket`

A coarse king-location category used in feature indexing. It lets the same
piece/square use different weights depending on king placement.

Actual Enyo NNUE value:

```text
king bucket ids = 32
king squares    = 64 squares mapped into those 32 ids
```

`weight`

A learned number stored in the network file. Training changes weights; engine
search reads fixed weights.

Actual Enyo NNUE trained values:

```text
input feature rows   = 24576
input row width      = 1024
input weights        = 24576 * 1024 = 25165824 int16 values
                     = 25,165,824 input weights
input biases         = 1024 int16 values
L1 weights           = 2048 * 16 = 32768 int8 values
L1 biases            = 16 int32 values
L2 weights           = 16 * 32 = 512 float values
L2 biases            = 32 float values
output weights       = 32 float values
output bias          = 1 float value
total trained values = 25200209
.nn payload size     = 50368836 bytes, about 48.0 MiB
```

`hidden neuron`

One learned numeric slot inside the network. Enyo NNUE has 1024 accumulator
slots per perspective.

Actual Enyo NNUE values:

```text
first hidden / accumulator width per perspective = 1024
perspectives                                     = 2
combined dense-head input                         = 2048
dense hidden layer 1                              = 16
dense hidden layer 2                              = 32
output                                            = 1 score
```

`accumulator`

The summed first hidden layer. Each active feature adds one 1024-wide row of
input weights into it.

Actual Enyo NNUE values:

```text
white perspective accumulator = 1024 values
black perspective accumulator = 1024 values
combined head input           = 2048 values
incremental update width      = 1024 values per changed feature row
```

`perspective`

The side whose king bucket and board view are used for feature indexing. Enyo
keeps both white and black perspective accumulators.

`us` / `them`

The side-to-move accumulator and the opponent accumulator as passed to the
dense head.

`dense head`

The small non-incremental part after the accumulator. It combines
`us[1024] + them[1024]` into one centipawn score.

Actual Enyo NNUE values:

```text
input          = 2048 int8-clipped accumulator values
L1 / layer 1   = 16 values
L2 / layer 2   = 32 values
output weights = 32 values
output bias    = 1 value
output         = 1 centipawn score
```

`L1`

Layer 1 of the dense head. It is the first small calculation stage after the
incremental accumulator. It reads the 2048 combined accumulator values and
produces 16 values.

Actual Enyo NNUE values:

```text
L1 input values  = 2048
L1 output values = 16
L1 weights       = 2048 * 16 = 32768 int8 values
L1 biases        = 16 int32 values
```

`L2`

Layer 2 of the dense head. It is the next small calculation stage. It reads the
16 L1 values and produces 32 values for the final scoring layer.

Actual Enyo NNUE values:

```text
L2 input values  = 16
L2 output values = 32
L2 weights       = 16 * 32 = 512 float values
L2 biases        = 32 float values
```

`output layer`

The final dense-head calculation. It reads the 32 L2 values and produces the
single centipawn score used by search.

Actual Enyo NNUE values:

```text
output input values = 32
output scores       = 1
output weights      = 32 float values
output bias         = 1 float value
```

`packed tensor`

Numeric training arrays produced from JSONL rows. They are training data, not a
trained net.

`label`

The training target attached to a position, usually a teacher centipawn score.

`teacher`

The engine or source that supplies target scores, usually Stockfish.

`batch`

Many training rows processed together in one optimizer step.

`epoch`

One pass over the sampled training rows.

`loss`

The training error PyTorch minimizes.

`Huber`

A centipawn loss that behaves like squared error for small mistakes and closer
to absolute error for large mistakes.

`MPE`

Mean probability error. In this project it means a WDL-shaped objective or
metric.

`target clamp`

A maximum absolute teacher score used for training, for example clamping
`+1800` to `+1000`.

`SPRT`

The engine match test used to decide whether a candidate net is stronger in
actual play.

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

## What The NNUE Is

The NNUE is a fast position-evaluation function used by Enyo search. It takes a
chess position and returns one centipawn score.

It is not a move picker by itself. Search still chooses moves. The NNUE changes
how leaf positions are evaluated inside alpha-beta search.

Conceptually:

```text
FEN position
  -> active sparse NNUE features
  -> accumulator values
  -> small dense network head
  -> centipawn score
  -> search uses that score
```

Training changes the stored numeric weights in the network. During a game those
weights are fixed; only the accumulator values change as pieces move.

## Features, Buckets, Neurons, Weights

### Feature

A feature is one possible board fact used as NNUE input. It is not a value by
itself; it is an index into the first weight table.

For Enyo NNUE, a feature is roughly:

```text
perspective + king bucket + piece/color type + piece square
```

The runtime feature address is:

```text
feature =
    king_bucket * 12 * 64
  + relative_piece_type * 64
  + mirrored_piece_square
```

Current shape:

```text
king buckets      = 32
piece/color types = 12
squares           = 64
input features    = 32 * 12 * 64 = 24576
```

### Active Feature

An active feature is a feature row that is used for the current FEN.

The network has 24576 possible feature rows, but most are inactive for any one
position. Enyo walks the board, finds every piece, and creates one feature row
per piece for each perspective.

In code terms, `enumerate_pieces()` in Enyo's `src/nnue_model_board.cpp`
emits:

```text
(piece_type + piece_color, square)
```

for every occupied square. Then `ResetAccumulator()` in
Enyo's `src/nnue_model.hpp` turns each entry into a feature index:

```text
feature = FeatureIdx(piece_type, piece_color, piece_square, view_king_square, view)
```

and adds that feature row:

```text
accumulator[view][0..1023] += input_weights[feature][0..1023]
```

Example with 32 pieces on the board:

```text
white view:
  32 active features -> add 32 different 1024-wide rows

black view:
  32 active features -> add 32 different 1024-wide rows
```

Example after many captures with 10 pieces left:

```text
white view:
  10 active features

black view:
  10 active features
```

Empty squares have no active feature. A missing captured piece has no active
feature. This is why incremental update is cheap: a quiet move usually removes
one old active feature and adds one new active feature.

### King Bucket

A king bucket is a coarse category for king location. It lets the same piece on
the same square mean different things depending on king placement.

Example:

```text
black knight on f5, white king in bucket 3
black knight on f5, white king in bucket 9
```

Those are different features and therefore use different learned weights.

A king bucket is not a hidden neuron. It is part of the feature address.

### Weight

A weight is one learned number stored in the network file.

The first layer has one 1024-wide weight row per input feature:

```text
input_weights[feature] = [w0, w1, w2, ..., w1023]
```

Training changes these numbers. Engine search only reads them.

### Hidden Neuron

A hidden neuron is a learned numeric slot inside the network. Enyo has 1024
accumulator slots per perspective.

These slots do not have fixed human meanings like "king safety" or "hanging
queen". A chess idea is usually spread across many weights and many hidden
slots.

### Accumulator

The accumulator is the summed first hidden layer.

For one perspective:

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

If the active feature is "white-view black knight on f5 with white king in
bucket 3", then the engine adds that feature's 1024 learned numbers to the
white-perspective accumulator.

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
