# Enyo NNUE

`README.md` describes how to build and validate candidate nets. This file
describes the native Enyo NNUE runtime: file layout, feature indices,
accumulators, and evaluation.

## Current Shape

```text
input buckets: 16
features:      16 * 12 * 64 = 12,288
hidden width:  1,024
L1 input:      2 * 1024 = 2,048
L1 output:     16
L2 output:     32
final output:  scalar centipawn score
```

The large input matrix is:

```cpp
Network::INPUT_WEIGHTS[N_FEATURES * N_HIDDEN]
```

With the current constants:

```text
12,288 features * 1,024 hidden = 12,582,912 int16 weights
```

`N_KING_BUCKETS` is compiled as the maximum supported count, but
`Network::INPUT_BUCKETS` is detected from the `.nn` size at load time. The
current production native net uses 16 input buckets.

## File Layout

The `.nn` file is a raw binary blob. There is no header or magic value. The
loader accepts known sizes and then points the runtime arrays at the loaded
storage.

Current 16-bucket file order:

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

Runtime names:

```text
file concept    C++ runtime name
input_weights   Network::INPUT_WEIGHTS
input_biases    Network::INPUT_BIASES
l1_weights      Network::L1_WEIGHTS
l1_biases       Network::L1_BIASES
l2_weights      Network::L2_WEIGHTS
l2_biases       Network::L2_BIASES
output_weights  Network::OUTPUT_WEIGHTS
output_bias     Network::OUTPUT_BIAS
```

In `nnue_model.cpp`, the backing storage is flat:

```cpp
alignas(64) int16_t s_input_weights[N_FEATURES * N_HIDDEN];
alignas(64) int16_t s_input_biases[N_HIDDEN];

Network::INPUT_WEIGHTS = s_input_weights;
Network::INPUT_BIASES  = s_input_biases;
```

The logical 2D address:

```cpp
const int16_t weight =
    Network::INPUT_WEIGHTS[feature * Network::N_HIDDEN + hidden];
```

That is the same as:

```cpp
input_weights[feature][hidden]
```

## Square And Feature Indexing

Enyo squares are h1-indexed:

```cpp
enum square : square_t {
    h1, g1, f1, e1, d1, c1, b1, a1,
    ...
};
```

So:

```text
h1 = 0
a1 = 7
a8 = 63
```

The network converts Enyo squares with:

```cpp
inline constexpr int to_net_sq(int enyo_sq) {
    return enyo_sq ^ 63;
}
```

A square alone is not a feature. A feature depends on:

```text
piece type
piece color
piece square
king square for the chosen view
view side
```

The runtime feature function is:

```cpp
int feature = Network::FeatureIdx(
    pt,           // enyo::PieceType
    piece_color,  // enyo::Color
    sq,           // enyo::square_t, h1-indexed
    king_sq,      // king square for this view
    view          // enyo::white or enyo::black
);
```

For example, "a white rook on a1" still needs the king square and the view:

```cpp
int f = Network::FeatureIdx(
    enyo::rook,
    enyo::white,
    enyo::a1,
    white_king_sq,
    enyo::white
);
```

To read the contribution of that feature to hidden neuron `h`:

```cpp
int16_t contribution =
    Network::INPUT_WEIGHTS[f * Network::N_HIDDEN + h];
```

So for hidden neuron `0`:

```cpp
int16_t a1_rook_h0 =
    Network::INPUT_WEIGHTS[f * Network::N_HIDDEN + 0];
```

## Build The Accumulator

The accumulator is the summed input layer for both perspectives:

```cpp
struct Network::Accumulator {
    acc_t values[2][N_HIDDEN]; // [perspective][hidden]
};
```

For one perspective, `ResetAccumulator` starts from the input biases and adds
one full 1024-wide row per active feature:

```cpp
void ResetAccumulator(Accumulator* dest,
                      enyo::Color view,
                      enyo::square_t king_sq,
                      const PieceEntry* pieces,
                      size_t npieces)
{
    acc_t* values = dest->values[view];

    std::memcpy(values, Network::INPUT_BIASES,
                sizeof(acc_t) * Network::N_HIDDEN);

    for (size_t i = 0; i < npieces; ++i) {
        int piece = pieces[i].piece_code;
        auto pt = static_cast<enyo::PieceType>((piece >> 1) + 1);
        auto pc = static_cast<enyo::Color>(piece & 1);

        int feature = Network::FeatureIdx(
            pt, pc, pieces[i].sq, king_sq, view);

        for (int h = 0; h < Network::N_HIDDEN; ++h)
            values[h] += Network::INPUT_WEIGHTS[
                feature * Network::N_HIDDEN + h];
    }
}
```

The real code batches those feature additions through `Network::Delta` and
`Network::ApplyDelta`, but the math is exactly the loop above.

For a normal position, the engine needs two rows:

```cpp
Network::ResetAccumulator(&acc, enyo::white, white_king_sq, pieces, npieces);
Network::ResetAccumulator(&acc, enyo::black, black_king_sq, pieces, npieces);
```

## Incremental Updates

During search, Enyo does not rebuild both accumulators at every node. It keeps:

```cpp
network_accumulator_stack[currentAccumulator]
```

A quiet move usually removes one feature row and adds one feature row:

```cpp
delta.rem[delta.r++] = old_piece_feature;
delta.add[delta.a++] = new_piece_feature;

Network::ApplyDelta(dest_values, parent_values, &delta);
```

A capture removes the moved piece's old feature and the captured piece feature,
then adds the moved piece's new feature:

```cpp
delta.rem[delta.r++] = old_piece_feature;
delta.rem[delta.r++] = captured_piece_feature;
delta.add[delta.a++] = new_piece_feature;
```

A king move can change many feature addresses. If the king crosses the center
file or changes king bucket, `Network::MoveRequiresRefresh()` returns true and
the affected perspective is rebuilt from the board instead of updated by a
small delta.

`s_input_weights` is therefore loaded once, but it is not used only once. It is
used whenever Enyo builds, refreshes, adds, or subtracts input feature rows.

## Evaluation Path

Search calls:

```cpp
template <Color Us, bool UseNNUE = true>
Value evaluate(Board& b, NNUE::Net* nnue)
{
    if constexpr (UseNNUE) {
        if (Network::enabled && Network::INPUT_WEIGHTS != nullptr)
            return static_cast<Value>(nnue->Evaluate2(b, Us));

        return static_cast<Value>(nnue->Evaluate(Us));
    }

    return static_cast<Value>(enyo::HCE_evaluation<Us>(b));
}
```

Native NNUE evaluation is:

```cpp
int32_t Net::Evaluate2(enyo::Board& board, enyo::Color side)
{
    Network::Accumulator& accumulator =
        network_accumulator_stack[currentAccumulator];

    if (accumulator.eval_correct[side])
        return accumulator.eval[side];

    auto& entry = network_eval_cache[
        board.hash & (network_eval_cache_size - 1)];
    if (entry.hash == board.hash && entry.valid) {
        accumulator.eval[side] = entry.eval;
        accumulator.eval_correct[side] = 1;
        return accumulator.eval[side];
    }

    ensure_network(board);

    int score = Network::Propagate(&accumulator, static_cast<int>(side));
    int eval = Network::ScaleEval(board, score);

    accumulator.eval[side] = eval;
    accumulator.eval_correct[side] = 1;
    entry.hash = board.hash;
    entry.eval = eval;
    entry.valid = 1;
    return eval;
}
```

`ensure_network(board)` materializes the accumulator for the current ply. It
uses lazy deltas from the parent when possible, and refreshes from the board
when needed.

## Dense Forward Pass

`Network::Propagate()` turns the accumulator into a centipawn score:

```cpp
int Propagate(const Accumulator* acc, int stm)
{
    int8_t x0[N_L1];
    float  x1[N_L2];
    float  x2[N_L3];

    InputReLU(x0, acc, stm);
    L1AffineReLU(x1, x0);
    L2AffineReLU(x2, x1);
    return static_cast<int>(L3Transform(x2) / 32.0f);
}
```

`InputReLU` chooses side-to-move first:

```cpp
const int views[2] = {stm, !stm};

for (int v = 0; v < 2; ++v) {
    const acc_t* in = acc->values[views[v]];
    int8_t* out = &x0[N_HIDDEN * v];

    for (int h = 0; h < N_HIDDEN; ++h)
        out[h] = clamp(in[h], 0, 127 << QUANT1_BITS) >> QUANT1_BITS;
}
```

So the dense head sees:

```text
x0[0..1023]     = side-to-move accumulator
x0[1024..2047]  = opponent accumulator
```

Then:

```text
L1AffineReLU: 2048 int8  -> 16 float
L2AffineReLU:   16 float -> 32 float
L3Transform:    32 float -> 1 score
```

Finally `Network::ScaleEval()` applies material/phase scaling and clamps the
result:

```cpp
int scaled = (128 + phase) * score / 128;
return std::clamp(scaled, -2045, 2045);
```

## Training Notes

Training changes the weights. Search does not.

Training loop conceptually:

```text
FEN -> active features -> accumulator -> dense head -> predicted score
predicted score vs teacher score -> loss -> backprop -> update weights
```

Runtime conceptually:

```text
board -> active features/deltas -> accumulator -> dense head -> score
```

A training/export pipeline writes `model.nn`. The engine loads that file and
uses the stored arrays as fixed weights during search.
