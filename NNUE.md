# Enyo NNUE

```sh
• Current native Enyo NNUE:

  input buckets: 16
  features:      12,288
  hidden width:  1,024
  L1 input:      2 * 1024 = 2,048
  L1 output:     16
  L2 output:     32
  final output:  scalar

  So the big accumulator matrix is:

  12,288 features * 1,024 hidden = 12,582,912 int16 weights

  Plus:

  1,024 input biases
  2,048 * 16 int8 l1 weights
  16 l1 biases
  16 * 32 float32 l2 weights
  32 l2 biases
  32 output weights
  1 output bias

  current .nn size: 25,203,012 bytes.
```

`README.md` describes candidate creation and validation workflow. This file
describes the NNUE itself: file layout, feature rows, accumulators, training
mechanics, and runtime evaluation.

## Runtime Net Layout

Enyo's native `.nn` is a raw binary file. There is no header, magic, or embedded
architecture metadata. The loader detects the supported layout from the file
size.

For the current 16-input-bucket native net, the file is written in this order:

```text
input_weights   int16   [12288, 1024]
input_biases    int16   [1024]
l1_weights      int8    [16, 2048]
l1_biases       int32   [16]
l2_weights      float32 [32, 16]
l2_biases       float32 [32]
output_weights  float32 [32]
output_bias     float32 [1]
```

Names differ by layer:

```text
NNUE.md concept       Python loader              Bullet name   Enyo runtime
input_weights         Net.input_weights          l0w           s_input_weights
input_biases          Net.input_biases           l0b           s_input_biases
l1_weights            Net.l1_weights             l1w           s_l1_weights
l1_biases             Net.l1_biases              l1b           s_l1_biases
l2_weights            Net.l2_weights             l2w           s_l2_weights
l2_biases             Net.l2_biases              l2b           s_l2_biases
output_weights        Net.output_weights         l3w           s_output_weights
output_bias           Net.output_bias            l3b           s_output_bias
```

`s_input_weights` is declared as one flat array:

```cpp
int16_t s_input_weights[N_FEATURES * N_HIDDEN];
```

The logical two-dimensional address is:

```cpp
s_input_weights[feature * N_HIDDEN + hidden]
```

This is the element that would be `input_weights[feature][hidden]` in a 2D
matrix.

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

More precisely, a feature depends on:

```text
view side
piece type
piece color
piece square
king square for that view side
```

So a square such as `a1` is not a feature by itself. For example, "white rook on
`a1` from White's perspective" needs the white king square too. In Enyo square
indexing, `h1 = 0`, so `a1 = 7`.

One active feature row contains one weight for every hidden slot:

```cpp
int feature = feature_index(piece_type, piece_color, square, king_square, view);
int16_t weight = s_input_weights[feature * N_HIDDEN + hidden];
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

In runtime terms:

```cpp
for (int hidden = 0; hidden < N_HIDDEN; ++hidden) {
    accumulator[hidden] += s_input_weights[feature * N_HIDDEN + hidden];
}
```

The net weights are fixed during search. The accumulator is the per-position
sum of the currently active fixed feature rows.

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

## Runtime Evaluation Path

During search, Enyo evaluates the current board from the side-to-move
perspective. Positive means good for the side to move at that node.

The search entry point is:

```text
search.cpp evaluate<Us>()
```

It chooses the evaluation path in this order:

```text
if native Network is enabled and loaded:
    nnue->Evaluate2(board, side_to_move)
else if use_nnue is enabled:
    old NNUE::Evaluate()
else:
    HCE_evaluation()
```

For the native net, `Evaluate2` does:

```text
1. Reuse cached eval if this accumulator already has a valid score.
2. Reuse the board-hash eval cache if the same board was evaluated recently.
3. Ensure the current accumulator is materialized.
4. Run Network::Propagate(accumulator, side_to_move).
5. Apply material/phase scaling and clamp.
6. Cache and return the centipawn score.
```

`ensure_network(board)` is the bridge between search and NNUE. It makes sure
the accumulator for the current ply is correct. If a lazy move delta is enough,
it applies the delta from the parent accumulator. If the move crossed a king
bucket or the lazy chain is not usable, it refreshes the affected accumulator
from the board.

`Network::Propagate` is the dense forward pass:

```text
InputReLU:
    choose accumulator[side_to_move] as us
    choose accumulator[other_side] as them
    concatenate us + them -> 2048 values
    clamp to [0, 127 << 5]
    divide by 32 into int8 inputs

L1AffineReLU:
    2048 int8 inputs -> 16 float values
    add l1_biases
    ReLU

L2AffineReLU:
    16 float values -> 32 float values
    add l2_biases
    ReLU

L3Transform:
    dot 32 float values with output_weights
    add output_bias
    divide by 32
```

After `Propagate`, Enyo applies a phase scale:

```text
phase = 3 * minor_count + 5 * rook_count + 10 * queen_count
scaled_score = (128 + phase) * score / 128
final_score = clamp(scaled_score, -2045, 2045)
```

So evaluation is not "look up one neuron". A position activates many feature
rows, those rows build two 1024-wide accumulators, and the side-to-move
accumulator pair is passed through the dense head to produce one centipawn
score.

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

`s_input_weights` is not only used once at network initialization. It is loaded
once from the `.nn` file and then kept as fixed runtime storage. Search uses it
whenever it needs to:

```text
build an accumulator from scratch
refresh an accumulator after a king-bucket change
add or subtract feature rows for an incremental move update
```

What avoids work is the accumulator stack. Most search nodes start from the
parent accumulator and apply a small delta instead of summing every piece row
again.

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
