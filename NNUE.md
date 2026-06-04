# Enyo NNUE

`README.md` describes how to build and validate candidate nets. This file
describes the native Enyo NNUE runtime: file layout, feature indices,
accumulators, and evaluation.

## Current Shape

```text
king buckets:  16
features:      16 * 12 * 64 = 12,288
hidden width:  1,024
L1 input:      2 * 1024 = 2,048
L1 output:     16
L2 output:     32
final output:  scalar centipawn score
```

`N_KING_BUCKETS` is compiled as the maximum supported count (32), but
the actual count is detected from the `.nn` size at load time.

## File Layout

The `.nn` file is a raw binary blob with no header. Layout for 16 buckets:

```text
input_weights   int16   [12288, 1024]   Network::INPUT_WEIGHTS
input_biases    int16   [1024]          Network::INPUT_BIASES
l1_weights      int8    [16, 2048]      Network::L1_WEIGHTS
l1_biases       int32   [16]            Network::L1_BIASES
l2_weights      float32 [32, 16]        Network::L2_WEIGHTS
l2_biases       float32 [32]            Network::L2_BIASES
output_weights  float32 [32]            Network::OUTPUT_WEIGHTS
output_bias     float32 [1]             Network::OUTPUT_BIAS
```

`INPUT_WEIGHTS` is a flat array; the logical 2D address is:

```cpp
INPUT_WEIGHTS[feature * N_HIDDEN + h]  // same as input_weights[feature][h]
```

## Features and the Accumulator

A feature is an integer index computed from piece type, piece color, piece
square, the king square for the chosen view, and the view side:

```cpp
int f = Network::FeatureIdx(pt, piece_color, sq, king_sq, view);
```

The accumulator holds the summed hidden layer for both perspectives:

```cpp
struct Network::Accumulator {
    acc_t values[2][N_HIDDEN]; // [perspective][hidden_neuron], int16
};
```

## Initializing the Accumulator

`ResetAccumulator` sets one perspective to the input bias, then adds the
`INPUT_WEIGHTS` row for every active piece:

```cpp
// start from bias
memcpy(values[view], INPUT_BIASES, N_HIDDEN * sizeof(int16_t));

// add one weight row per piece on the board
for each piece (pt, pc, sq) on board:
    int f = Network::FeatureIdx(pt, pc, sq, king_sq, view);
    for (int h = 0; h < N_HIDDEN; ++h)
        values[view][h] += INPUT_WEIGHTS[f * N_HIDDEN + h];
```

Called twice at game start — once for each king square:

```cpp
Network::ResetAccumulator(&acc, enyo::white, white_king_sq, pieces, npieces);
Network::ResetAccumulator(&acc, enyo::black, black_king_sq, pieces, npieces);
```

## Incremental Updates

During search Enyo pushes/pops `network_accumulator_stack` rather than
rebuilding. A move changes only the features that differ from the parent.

### Quiet move: e2 to e4

The white pawn leaves e2 and arrives at e4. One feature index is removed
and one is added — per perspective:

```cpp
for (auto side : {enyo::white, enyo::black}) {
    auto king_sq = (side == enyo::white) ? white_king_sq : black_king_sq;

    int f_from = Network::FeatureIdx(enyo::pawn, enyo::white, enyo::e2, king_sq, side);
    int f_to   = Network::FeatureIdx(enyo::pawn, enyo::white, enyo::e4, king_sq, side);

    Network::ApplySubAdd(dest.values[side], src.values[side], f_from, f_to);
}
```

`ApplySubAdd` scalar form:

```cpp
for (int h = 0; h < N_HIDDEN; ++h)
    dest[h] = src[h]
            - INPUT_WEIGHTS[f_from * N_HIDDEN + h]
            + INPUT_WEIGHTS[f_to   * N_HIDDEN + h];
```

A capture adds a second removal (the captured piece's feature) before the
addition (`ApplySubSubAdd`). A king move that crosses the center file or
changes king bucket triggers a full perspective rebuild via the Finny table
cache (`network_refresh_table`), because all feature indices for that
perspective reference the king square.

## From Accumulator to Evaluation

`Network::Propagate()` runs the dense forward pass:

```
1. InputReLU:   acc.values[stm..!stm]  ->  x0: int8[2048]
2. L1AffineReLU: x0                    ->  x1: float[16]
3. L2AffineReLU: x1                    ->  x2: float[32]
4. L3Transform:  x2                    ->  raw_score: float
5. return int(raw_score / 32.0f)       // centipawns, stm-relative
```

**Step 1 — InputReLU** concatenates the side-to-move accumulator first,
then the opponent, and clamps each int16 value to `[0, 127]`:

```cpp
// x0[0..1023]    = stm  accumulator
// x0[1024..2047] = them accumulator
for (int h = 0; h < N_HIDDEN; ++h)
    x0[h] = clamp(acc.values[stm][h],  0, 127 << QUANT1_BITS) >> QUANT1_BITS;
    x0[N_HIDDEN + h] = clamp(acc.values[!stm][h], 0, 127 << QUANT1_BITS) >> QUANT1_BITS;
```

**Steps 2–4** are dense affine layers with ReLU, ending in a dot product
against `OUTPUT_WEIGHTS[output_bucket]`.

**Step 5 — ScaleEval** applies a material/phase multiplier and clamps:

```cpp
int scaled = (128 + phase) * score / 128;
return std::clamp(scaled, -2045, 2045);
```

## Training Notes

Training changes the weights. Search does not.

```text
FEN -> active features -> accumulator -> dense head -> predicted score
predicted score vs teacher score -> loss -> backprop -> update weights
```

The export pipeline writes `model.nn`. The engine loads it once and treats
all weight arrays as read-only during search.
