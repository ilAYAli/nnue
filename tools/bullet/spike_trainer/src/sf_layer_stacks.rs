//! The nnue-pytorch "layer stacks" network (features Full_Threats+HalfKAv2_hm^)
//! as trained by nnue-pytorch at ccc0838^ and evaluated by the engine's
//! Stockfish `current` feature variant:
//!
//! * feature transformer: Full_Threats rows (int8 export) followed by
//!   HalfKAv2_hm^ rows (12 piece planes while training, one shared 768-row
//!   virtual factor; 11 planes with merged kings in the export), L1 outputs
//!   plus 8 PSQT outputs per row. With input set Enyo HalfKA 32x12
//!   (architecture 26 input_set 1, see `enyo_halfka_index`) the king rows are
//!   Enyo's 32 buckets x 12 planes instead, trained and exported as they are,
//!   with the same 768-row virtual factor;
//! * per perspective clamp(acc, 0, 255/256), pairwise product of the two
//!   halves (L1/2 values), side to move first;
//! * 8 layer stacks picked by the bucket scheme (`BucketScheme`: Stockfish's
//!   (pieces - 1) / 4, or Enyo's game phase), which also picks the PSQT
//!   bucket: fc0 L1 -> L2 (factorised across stacks), [sqr(fc0), fc0]
//!   clipped to 127/128, fc1 2*L2 -> L3, [sqr(fc1), fc1] clipped, fc2 over
//!   [fc0 branch, fc1 branch] -> 1, plus fc0[L2-2] - fc0[L2-1] passthrough,
//!   plus (psqt_stm - psqt_nstm) / 2. L1 is the architecture-26 hidden (the
//!   engine loads 1024 and 1280), L2 its fc0_out (32 and 64), L3 its fc1_out.
//!
//! Everything here is CPU-side and graph independent: input indexing, the
//! nnue-pytorch initialisation, Stockfish .nnue export/import (nnue-pytorch
//! serialize.py byte layout), Enyo format-1 architecture-26 export/import
//! (tools/lib/enyo_net_v1.py layout), the nnue-pytorch data skip predicate
//! and the tablebase score remap. The Bullet graph lives in main.rs.

use std::{
    cell::Cell,
    collections::BTreeMap,
    fs,
    io::{self, Write},
    path::Path,
    sync::Mutex,
};

use bullet_lib::{
    game::{formats::bulletformat::ChessBoard, inputs::SparseInputType, outputs::OutputBuckets},
    value::loader::{
        sfbinpack::{PieceType, TrainingDataEntry},
        DataLoader,
    },
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sha2::{Digest, Sha256};

use crate::enyo_threats;

pub const THREAT_FEATURES: usize = enyo_threats::DIMENSIONS;
pub const KING_BUCKETS: usize = 32;
pub const PLANES: usize = 12 * 64;
pub const EXPORT_PLANES: usize = 11 * 64;
/// Training rows of either input set (32 king buckets x 12 planes).
pub const HALFKA_FEATURES: usize = KING_BUCKETS * PLANES;
pub const HALFKA_EXPORT_FEATURES: usize = KING_BUCKETS * EXPORT_PLANES;
pub const INPUTS: usize = THREAT_FEATURES + HALFKA_FEATURES;
pub const STACKS: usize = 8;
pub const PSQT_BUCKETS: usize = 8;

// nnue-pytorch QuantizationConfig defaults.
pub const NNUE2SCORE: f32 = 600.0;
pub const FT_ONE: f32 = 256.0;
pub const FT_MAX: f32 = 255.0 / 256.0;
pub const PSQT_SCALE: f32 = NNUE2SCORE * 16.0;
pub const HIDDEN_ONE: f32 = 128.0;
pub const HIDDEN_MAX: f32 = 127.0 / 128.0;
pub const FC0_SCALE: f32 = 128.0;
pub const FC1_SCALE: f32 = 64.0;
pub const FC2_SCALE: f32 = 128.0;
pub const THREAT_MAX: f32 = 127.0 / 256.0;

pub const VERSION: u32 = 0x6A44_8AFA;
pub const FULL_THREATS_HASH: u32 = 0x2E6B_9D04;
pub const HALFKA_HASH: u32 = 0x7F23_4CB8;
pub const DEFAULT_DESCRIPTION: &str = "Network trained with the Enyo Bullet layer-stacks trainer.";
const LEB128_MAGIC: &[u8] = b"COMPRESSED_LEB128";

/// The king-relative piece-square inputs of the feature transformer (the
/// architecture-26 layout field input_set).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InputSet {
    /// Stockfish HalfKAv2_hm: 32 mirrored king buckets x 11 planes (kings
    /// merged) in the export. input_set 0.
    #[default]
    HalfKav2Hm,
    /// Enyo HalfKA: 32 king buckets x 12 planes, the layout of Enyo's
    /// architecture-24/25 nets. input_set 1; no Stockfish .nnue form.
    EnyoHalfKa32x12,
}

impl InputSet {
    pub fn from_name(name: &str) -> Result<InputSet, String> {
        match name {
            "halfkav2_hm" => Ok(InputSet::HalfKav2Hm),
            "enyo_halfka_32x12" => Ok(InputSet::EnyoHalfKa32x12),
            other => Err(format!("unknown input_set {other} (halfkav2_hm or enyo_halfka_32x12)")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            InputSet::HalfKav2Hm => "halfkav2_hm",
            InputSet::EnyoHalfKa32x12 => "enyo_halfka_32x12",
        }
    }

    /// The architecture-26 layout value.
    pub fn layout_id(self) -> u32 {
        match self {
            InputSet::HalfKav2Hm => 0,
            InputSet::EnyoHalfKa32x12 => 1,
        }
    }

    /// Training row of a piece for one perspective; see `halfka_index` and
    /// `enyo_halfka_index`.
    pub fn index(self, piece_type: usize, own: bool, square: usize, king: usize) -> usize {
        match self {
            InputSet::HalfKav2Hm => halfka_index(piece_type, own, square, king),
            InputSet::EnyoHalfKa32x12 => enyo_halfka_index(piece_type, own, square, king),
        }
    }

    /// The piece type (0 pawn .. 5 king) and ownership of a training plane.
    fn plane_piece(self, plane: usize) -> (usize, bool) {
        match self {
            InputSet::HalfKav2Hm => (plane / 2, plane % 2 == 0),
            InputSet::EnyoHalfKa32x12 => (plane % 6, plane < 6),
        }
    }

    /// King rows in the exported net.
    pub fn export_rows(self) -> usize {
        match self {
            InputSet::HalfKav2Hm => HALFKA_EXPORT_FEATURES,
            InputSet::EnyoHalfKa32x12 => HALFKA_FEATURES,
        }
    }

    /// The training row an exported row comes from.
    fn export_source_row(self, row: usize) -> usize {
        match self {
            InputSet::HalfKav2Hm => export_halfka_row(row),
            InputSet::EnyoHalfKa32x12 => row,
        }
    }
}

/// What picks the layer stack and the PSQT bucket of a position (the
/// architecture-26 layout field bucket_scheme; the engine's
/// NNUE::Stockfish::LayerStackBucket).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BucketScheme {
    /// Stockfish: (pieces - 1) / 4, kings and pawns counted. bucket_scheme 0;
    /// the only scheme a Stockfish .nnue holds.
    #[default]
    PieceCount,
    /// Enyo game phase: phase = N + B + 2 R + 4 Q over both sides (24 at the
    /// start, clamped to 24 after promotions), bucket = phase * 8 / 25, so
    /// phases 0-3 are bucket 0 and every further three points one bucket up.
    /// bucket_scheme 1.
    Phase,
}

impl BucketScheme {
    pub fn from_name(name: &str) -> Result<BucketScheme, String> {
        match name {
            "piece_count" => Ok(BucketScheme::PieceCount),
            "phase" => Ok(BucketScheme::Phase),
            other => Err(format!("unknown bucket_scheme {other} (piece_count or phase)")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            BucketScheme::PieceCount => "piece_count",
            BucketScheme::Phase => "phase",
        }
    }

    /// The architecture-26 layout value.
    pub fn layout_id(self) -> u32 {
        match self {
            BucketScheme::PieceCount => 0,
            BucketScheme::Phase => 1,
        }
    }

    /// Layer-stack and PSQT bucket of a position.
    pub fn bucket(self, pos: &ChessBoard) -> usize {
        match self {
            BucketScheme::PieceCount => stack_index(pos.occ().count_ones()),
            BucketScheme::Phase => phase_index(pos),
        }
    }
}

/// Layer widths, the king input set and the bucket scheme. The engine loads
/// L1 (hidden) 1024 or 1280, L2 (fc0_out) 32 or 64 and L3 32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub l1: usize,
    pub l2: usize,
    pub l3: usize,
    pub input_set: InputSet,
    pub bucket_scheme: BucketScheme,
}

impl Spec {
    pub const STOCKFISH: Spec =
        Spec { l1: 1024, l2: 32, l3: 32, input_set: InputSet::HalfKav2Hm, bucket_scheme: BucketScheme::PieceCount };
    /// The accumulator widths (L1, hidden) the engine loads.
    pub const ENGINE_L1: [usize; 2] = [1024, 1280];
    /// The fc0 widths (L2) the engine loads.
    pub const ENGINE_L2: [usize; 2] = [32, 64];

    /// Widths the engine loads (1024|1280/32|64/32), whatever the input set
    /// and bucket scheme.
    pub fn engine_widths(self) -> bool {
        Self::ENGINE_L1.contains(&self.l1) && Self::ENGINE_L2.contains(&self.l2) && self.l3 == Self::STOCKFISH.l3
    }

    /// Whether a run publishes the net as a Stockfish .nnue: HalfKAv2_hm with
    /// piece-count buckets, except the engine-loadable variants other than
    /// 1024/32/32, which the engine loads only as architecture 26.
    pub fn nnue_form(self) -> bool {
        self.input_set == InputSet::HalfKav2Hm
            && self.bucket_scheme == BucketScheme::PieceCount
            && ((self.l1, self.l2) == (Self::STOCKFISH.l1, Self::STOCKFISH.l2) || !self.engine_widths())
    }

    pub fn validate(self) -> Result<(), String> {
        if self.l1 == 0 || self.l1 % 64 != 0 {
            return Err(format!("layer-stacks hidden (L1) must be a positive multiple of 64, got {}", self.l1));
        }
        if self.l2 < 2 || self.l3 == 0 {
            return Err(format!("layer-stacks needs l2_size >= 2 and l3_size >= 1, got {}/{}", self.l2, self.l3));
        }
        Ok(())
    }

    /// (id, rows, cols) of every Bullet weight, column-major as Bullet
    /// stores it: element (row, col) at col * rows + row.
    pub fn weight_shapes(self) -> Vec<(&'static str, usize, usize)> {
        let Spec { l1, l2, l3, .. } = self;
        vec![
            ("l0t", l1, THREAT_FEATURES),
            ("l0h", l1, HALFKA_FEATURES),
            ("l0v", l1, PLANES),
            ("l0b", l1, 1),
            ("p0t", PSQT_BUCKETS, THREAT_FEATURES),
            ("p0h", PSQT_BUCKETS, HALFKA_FEATURES),
            ("p0v", PSQT_BUCKETS, PLANES),
            ("l1w", STACKS * l2, l1),
            ("l1b", STACKS * l2, 1),
            ("l1f", l2, l1),
            ("l1fb", l2, 1),
            ("l2w", STACKS * l3, 2 * l2),
            ("l2b", STACKS * l3, 1),
            ("l3w", STACKS, 2 * l2 + 2 * l3),
            ("l3b", STACKS, 1),
        ]
    }

    pub fn feature_hash(self) -> u32 {
        FULL_THREATS_HASH.rotate_left(1) ^ HALFKA_HASH
    }

    pub fn fc_hash(self) -> u32 {
        let mut previous = 0xEC42_E90D_u32 ^ (self.l1 as u32 * 2);
        for outputs in [self.l2, self.l3, 1] {
            let mut hash = 0xCC03_DAE4_u32.wrapping_add(outputs as u32);
            hash ^= previous >> 1;
            hash ^= previous << 31;
            if outputs != 1 {
                hash = hash.wrapping_add(0x538D_24C7);
            }
            previous = hash;
        }
        previous
    }

    pub fn network_hash(self) -> u32 {
        self.fc_hash() ^ self.feature_hash() ^ (self.l1 as u32 * 2)
    }
}

pub type Weights = BTreeMap<String, Vec<f32>>;

// ---------------------------------------------------------------------------
// Inputs and buckets

/// nnue-pytorch KingBuckets, indexed by the oriented king square (files e-h).
fn king_bucket(oriented_king: usize) -> usize {
    debug_assert!(oriented_king % 8 >= 4);
    31 - 4 * (oriented_king / 8) - (oriented_king % 8 - 4)
}

/// The oriented king square of a king bucket (nnue-pytorch InverseKingBuckets).
pub fn bucket_king_square(bucket: usize) -> usize {
    let k = 31 - bucket;
    8 * (k / 4) + 4 + k % 4
}

/// HalfKAv2_hm^ training index (12 planes) for one perspective. `square` and
/// `king` are already in that perspective's orientation (a1 = 0, own pieces
/// at the bottom); `own` tells whether the piece belongs to the perspective.
pub fn halfka_index(piece_type: usize, own: bool, square: usize, king: usize) -> usize {
    let mirror = if king % 8 < 4 { 7 } else { 0 };
    let plane = 2 * piece_type + usize::from(!own);
    (square ^ mirror) + 64 * plane + PLANES * king_bucket(king ^ mirror)
}

/// Enyo king buckets (after Berserk), indexed by the oriented king square
/// with a8 = 0 from the perspective's side: the engine's
/// src/nnue/enyo/enyo_halfka.cpp and the architecture-24/25 table.
#[rustfmt::skip]
const ENYO_KING_BUCKETS: [usize; 64] = [
    31, 30, 29, 28, 28, 29, 30, 31,
    27, 26, 25, 24, 24, 25, 26, 27,
    23, 22, 21, 20, 20, 21, 22, 23,
    19, 18, 17, 16, 16, 17, 18, 19,
    15, 14, 13, 12, 12, 13, 14, 15,
    11, 10,  9,  8,  8,  9, 10, 11,
     7,  6,  5,  4,  4,  5,  6,  7,
     3,  2,  1,  0,  0,  1,  2,  3,
];

/// Enyo HalfKA 32x12 index (architecture-26 input_set 1, identical to the
/// architecture-24/25 FeatureIdx) for one perspective, with `square` and
/// `king` as for `halfka_index`. The board is seen from the perspective with
/// a8 = 0 (square ^ 56) and mirrored so the king stands on files e-h:
/// bucket * 768 + plane * 64 + oriented square, planes own pawn..king 0-5,
/// opponent pawn..king 6-11, buckets 0-3 on the back rank (file e..h), +4 per
/// rank.
pub fn enyo_halfka_index(piece_type: usize, own: bool, square: usize, king: usize) -> usize {
    let orient = 56 ^ if king % 8 < 4 { 7 } else { 0 };
    let plane = 6 * usize::from(!own) + piece_type;
    PLANES * ENYO_KING_BUCKETS[king ^ orient] + 64 * plane + (square ^ orient)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LayerStackInputs {
    pub input_set: InputSet,
}

impl SparseInputType for LayerStackInputs {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        INPUTS
    }

    fn max_active(&self) -> usize {
        enyo_threats::MAX_ACTIVE + 32
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &ChessBoard, mut f: F) {
        // Bullet boards are side-to-move relative: a1 = 0 from the side to
        // move, colour bit clear for its pieces, opp_ksq already flipped.
        let threats = enyo_threats::active_features(pos);
        assert_eq!(threats[0].len(), threats[1].len(), "FullThreats perspective feature counts differ");
        for i in 0..threats[0].len() {
            f(threats[0].get(i), threats[1].get(i));
        }
        let stm_king = usize::from(pos.our_ksq());
        let nstm_king = usize::from(pos.opp_ksq());
        let set = self.input_set;
        for (piece, square) in pos.into_iter() {
            let piece_type = usize::from(piece & 7);
            let ours = piece & 8 == 0;
            let square = usize::from(square);
            f(
                THREAT_FEATURES + set.index(piece_type, ours, square, stm_king),
                THREAT_FEATURES + set.index(piece_type, !ours, square ^ 56, nstm_king),
            );
        }
    }

    fn shorthand(&self) -> String {
        match self.input_set {
            InputSet::HalfKav2Hm => "full-threats+halfkav2hm".to_string(),
            InputSet::EnyoHalfKa32x12 => "full-threats+enyo-halfka-32x12".to_string(),
        }
    }

    fn description(&self) -> String {
        match self.input_set {
            InputSet::HalfKav2Hm => "nnue-pytorch Full_Threats+HalfKAv2_hm^ (12-plane training layout)".to_string(),
            InputSet::EnyoHalfKa32x12 => "Full_Threats + Enyo HalfKA 32 king buckets x 12 planes".to_string(),
        }
    }
}

/// Piece-count bucket: (pieces - 1) / 4.
pub fn stack_index(pieces: u32) -> usize {
    ((pieces.max(1) - 1) / 4).min(STACKS as u32 - 1) as usize
}

/// Phase bucket (`BucketScheme::Phase`): min(7, phase * 8 / 25) with
/// phase = min(24, N + B + 2 R + 4 Q) over both sides.
pub fn phase_index(pos: &ChessBoard) -> usize {
    // Bullet piece types: pawn 0, knight 1, bishop 2, rook 3, queen 4, king 5.
    const PHASE: [u32; 8] = [0, 1, 1, 2, 4, 0, 0, 0];
    let phase: u32 = pos.into_iter().map(|(piece, _)| PHASE[usize::from(piece & 7)]).sum();
    (phase.min(24) * 8 / 25).min(STACKS as u32 - 1) as usize
}

/// Bullet's output buckets: the layer stack (and PSQT bucket) of a position.
#[derive(Clone, Copy, Debug, Default)]
pub struct LayerStackBuckets {
    pub scheme: BucketScheme,
}

impl OutputBuckets<ChessBoard> for LayerStackBuckets {
    const BUCKETS: usize = STACKS;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        self.scheme.bucket(pos) as u8
    }
}

// ---------------------------------------------------------------------------
// Initialisation (nnue-pytorch defaults)

fn uniform(rng: &mut ChaCha8Rng, len: usize, bound: f32) -> Vec<f32> {
    (0..len).map(|_| rng.random_range(-bound..bound)).collect()
}

/// nnue-pytorch initial values, seeded: FT rows uniform(+-1/sqrt(rows of
/// their feature set)), FT bias uniform(+-1/sqrt(all rows)), HalfKA PSQT
/// columns at the classical piece values / 600, threat PSQT zero, virtual
/// rows zero; dense layers torch.nn.Linear defaults with identical stacks,
/// output bias and the fc0 factoriser zero.
pub fn initial_weights(spec: Spec, seed: u64) -> Weights {
    let Spec { l1, l2, l3, input_set, .. } = spec;
    let mut weights = Weights::new();
    let mut tensor = |id: &str, values: Vec<f32>| {
        weights.insert(id.to_string(), values);
    };
    let rng = |id: &str| {
        let mut hash = Sha256::new();
        hash.update(seed.to_le_bytes());
        hash.update(b"layer-stacks/");
        hash.update(id.as_bytes());
        ChaCha8Rng::from_seed(hash.finalize().into())
    };

    tensor("l0t", uniform(&mut rng("l0t"), l1 * THREAT_FEATURES, (1.0 / THREAT_FEATURES as f32).sqrt()));
    tensor("l0h", uniform(&mut rng("l0h"), l1 * HALFKA_FEATURES, (1.0 / HALFKA_FEATURES as f32).sqrt()));
    tensor("l0v", vec![0.0; l1 * PLANES]);
    tensor("l0b", uniform(&mut rng("l0b"), l1, (1.0 / INPUTS as f32).sqrt()));
    tensor("p0t", vec![0.0; PSQT_BUCKETS * THREAT_FEATURES]);
    let mut psqt = vec![0.0; PSQT_BUCKETS * HALFKA_FEATURES];
    const PIECE_VALUES: [f32; 5] = [126.0, 781.0, 825.0, 1276.0, 2538.0];
    for feature in 0..HALFKA_FEATURES {
        let (piece_type, own) = input_set.plane_piece((feature % PLANES) / 64);
        if piece_type < 5 {
            let sign = if own { 1.0 } else { -1.0 };
            let value = sign * PIECE_VALUES[piece_type] / NNUE2SCORE;
            psqt[feature * PSQT_BUCKETS..(feature + 1) * PSQT_BUCKETS].fill(value);
        }
    }
    tensor("p0h", psqt);
    tensor("p0v", vec![0.0; PSQT_BUCKETS * PLANES]);

    // Bullet stores (rows = stacks * outputs, cols = inputs) column-major;
    // every stack starts from the same uniform draw as nnue-pytorch's
    // StackedLinear._init_uniformly.
    let stacked = |rng: &mut ChaCha8Rng, inputs: usize, outputs: usize| {
        let bound = (1.0 / inputs as f32).sqrt();
        let base = uniform(rng, inputs * outputs, bound);
        let bias = uniform(rng, outputs, bound);
        let mut w = vec![0.0; inputs * outputs * STACKS];
        for i in 0..inputs {
            for s in 0..STACKS {
                let start = i * outputs * STACKS + s * outputs;
                w[start..start + outputs].copy_from_slice(&base[i * outputs..(i + 1) * outputs]);
            }
        }
        (w, bias.repeat(STACKS))
    };
    let (w, b) = stacked(&mut rng("l1"), l1, l2);
    tensor("l1w", w);
    tensor("l1b", b);
    tensor("l1f", vec![0.0; l2 * l1]);
    tensor("l1fb", vec![0.0; l2]);
    let (w, b) = stacked(&mut rng("l2"), 2 * l2, l3);
    tensor("l2w", w);
    tensor("l2b", b);
    let (w, _) = stacked(&mut rng("l3"), 2 * l2 + 2 * l3, 1);
    tensor("l3w", w);
    tensor("l3b", vec![0.0; STACKS]);
    weights
}

// ---------------------------------------------------------------------------
// Bullet weight files

pub fn write_weights(path: &Path, weights: &Weights) -> io::Result<()> {
    let mut out = io::BufWriter::new(fs::File::create(path)?);
    for (id, values) in weights {
        out.write_all(id.as_bytes())?;
        out.write_all(b"\n")?;
        out.write_all(&values.len().to_le_bytes())?;
        for value in values {
            out.write_all(&value.to_le_bytes())?;
        }
    }
    out.flush()
}

pub fn read_weights(path: &Path) -> Result<Weights, String> {
    let bytes = fs::read(path).map_err(|err| format!("cannot read {}: {err}", path.display()))?;
    let mut weights = Weights::new();
    let mut offset = 0;
    const USIZE: usize = std::mem::size_of::<usize>();
    while offset < bytes.len() {
        let newline = bytes[offset..]
            .iter()
            .position(|&b| b == b'\n')
            .ok_or_else(|| format!("{}: truncated weight id", path.display()))?;
        let id = String::from_utf8_lossy(&bytes[offset..offset + newline]).into_owned();
        offset += newline + 1;
        let len = usize::from_le_bytes(
            bytes
                .get(offset..offset + USIZE)
                .ok_or_else(|| format!("{}: truncated length of {id}", path.display()))?
                .try_into()
                .unwrap(),
        );
        offset += USIZE;
        let raw = bytes
            .get(offset..offset + 4 * len)
            .ok_or_else(|| format!("{}: truncated values of {id}", path.display()))?;
        weights.insert(id, raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect());
        offset += 4 * len;
    }
    Ok(weights)
}

fn tensor<'a>(spec: Spec, weights: &'a Weights, id: &str) -> Result<&'a [f32], String> {
    let (_, rows, cols) = spec
        .weight_shapes()
        .into_iter()
        .find(|(name, _, _)| *name == id)
        .ok_or_else(|| format!("unknown layer-stacks tensor {id}"))?;
    let values = weights.get(id).ok_or_else(|| format!("missing layer-stacks tensor {id}"))?;
    if values.len() != rows * cols {
        return Err(format!("tensor {id} has {} values, expected {}", values.len(), rows * cols));
    }
    Ok(values)
}

// ---------------------------------------------------------------------------
// Stockfish .nnue export / import (nnue-pytorch model/utils/serialize.py)

fn quantise(value: f32, scale: f32, min: i64, max: i64, what: &str) -> Result<i64, String> {
    let q = (value * scale).round_ties_even() as i64;
    if q < min || q > max {
        return Err(format!("{what}: {value} quantises to {q}, outside [{min}, {max}]"));
    }
    Ok(q)
}

fn leb128_varints(values: impl IntoIterator<Item = i64>) -> Vec<u8> {
    let mut encoded = Vec::new();
    for mut value in values {
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if (value == 0 && byte & 0x40 == 0) || (value == -1 && byte & 0x40 != 0) {
                encoded.push(byte);
                break;
            }
            encoded.push(byte | 0x80);
        }
    }
    encoded
}

/// Stockfish's LEB128 block: magic, u32 byte count, varints.
pub fn encode_leb128(values: impl IntoIterator<Item = i64>, out: &mut Vec<u8>) {
    let encoded = leb128_varints(values);
    out.extend_from_slice(LEB128_MAGIC);
    out.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
    out.extend_from_slice(&encoded);
}

/// The format-1 LEB128 block: u32 byte count, varints.
fn encode_leb128_sized(values: impl IntoIterator<Item = i64>, out: &mut Vec<u8>) {
    let encoded = leb128_varints(values);
    out.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
    out.extend_from_slice(&encoded);
}

/// Merged HalfKA rows (12 planes, factor added) remapped to the 11-plane
/// export: the own king appears only on its bucket's square, the opponent
/// king fills the rest of the shared king plane.
fn export_halfka_row(row: usize) -> usize {
    let bucket = row / EXPORT_PLANES;
    let within = row % EXPORT_PLANES;
    let base = bucket * PLANES;
    if within < 640 {
        return base + within;
    }
    let square = within - 640;
    if square == bucket_king_square(bucket) {
        base + 640 + square
    } else {
        base + 704 + square
    }
}

/// One layer stack on its export grid; weights [output][input], unpadded.
#[derive(Clone, Debug, PartialEq)]
struct QuantStack {
    fc0_biases: Vec<i64>,
    fc0_weights: Vec<i64>,
    fc1_biases: Vec<i64>,
    fc1_weights: Vec<i64>,
    fc2_bias: i64,
    fc2_weights: Vec<i64>,
}

/// The integer net both export formats hold: feature rows in export order
/// (`InputSet::export_rows` king rows), then the layer stacks.
#[derive(Clone, Debug, PartialEq)]
struct QuantNet {
    ft_biases: Vec<i64>,
    threat_weights: Vec<i64>,
    threat_psqt: Vec<i64>,
    halfka_weights: Vec<i64>,
    halfka_psqt: Vec<i64>,
    stacks: Vec<QuantStack>,
}

const I8_RANGE: (i64, i64) = (-(i8::MAX as i64), i8::MAX as i64);
const I16_RANGE: (i64, i64) = (-(i16::MAX as i64), i16::MAX as i64);
const I32_RANGE: (i64, i64) = (-(i32::MAX as i64), i32::MAX as i64);

/// Bullet weights on the export grid: factors coalesced, kings merged for
/// HalfKAv2_hm.
fn quantise_net(spec: Spec, weights: &Weights) -> Result<QuantNet, String> {
    spec.validate()?;
    let Spec { l1, l2, l3, input_set, .. } = spec;
    let q_all = |values: &[f32], scale: f32, range: (i64, i64), what: &str| {
        values.iter().map(|&v| quantise(v, scale, range.0, range.1, what)).collect::<Result<Vec<_>, _>>()
    };

    let ft_biases = q_all(tensor(spec, weights, "l0b")?, FT_ONE, I16_RANGE, "ft bias")?;
    let threat_weights = q_all(tensor(spec, weights, "l0t")?, FT_ONE, I8_RANGE, "threat weight")?;
    let threat_psqt = q_all(tensor(spec, weights, "p0t")?, PSQT_SCALE, I32_RANGE, "threat psqt")?;

    // King rows: coalesce the factor (and merge kings for HalfKAv2_hm).
    let l0h = tensor(spec, weights, "l0h")?;
    let l0v = tensor(spec, weights, "l0v")?;
    let p0h = tensor(spec, weights, "p0h")?;
    let p0v = tensor(spec, weights, "p0v")?;
    let rows = input_set.export_rows();
    let mut halfka_weights = Vec::with_capacity(rows * l1);
    let mut halfka_psqt = Vec::with_capacity(rows * PSQT_BUCKETS);
    for row in 0..rows {
        let source = input_set.export_source_row(row);
        let plane = source % PLANES;
        for o in 0..l1 {
            let w = l0h[source * l1 + o] + l0v[plane * l1 + o];
            halfka_weights.push(quantise(w, FT_ONE, I16_RANGE.0, I16_RANGE.1, "halfka weight")?);
        }
        for o in 0..PSQT_BUCKETS {
            let w = p0h[source * PSQT_BUCKETS + o] + p0v[plane * PSQT_BUCKETS + o];
            halfka_psqt.push(quantise(w, PSQT_SCALE, I32_RANGE.0, I32_RANGE.1, "halfka psqt")?);
        }
    }

    let l1w = tensor(spec, weights, "l1w")?;
    let l1b = tensor(spec, weights, "l1b")?;
    let l1f = tensor(spec, weights, "l1f")?;
    let l1fb = tensor(spec, weights, "l1fb")?;
    let l2w = tensor(spec, weights, "l2w")?;
    let l2b = tensor(spec, weights, "l2b")?;
    let l3w = tensor(spec, weights, "l3w")?;
    let l3b = tensor(spec, weights, "l3b")?;
    let mut stacks = Vec::with_capacity(STACKS);
    for stack in 0..STACKS {
        let mut s = QuantStack {
            fc0_biases: Vec::with_capacity(l2),
            fc0_weights: Vec::with_capacity(l2 * l1),
            fc1_biases: Vec::with_capacity(l3),
            fc1_weights: Vec::with_capacity(l3 * 2 * l2),
            fc2_bias: 0,
            fc2_weights: Vec::with_capacity(2 * l2 + 2 * l3),
        };
        // fc0: the graph clamps the merged weight to the int8 grid range.
        for o in 0..l2 {
            let b = l1b[stack * l2 + o] + l1fb[o];
            s.fc0_biases.push(quantise(b, FC0_SCALE * HIDDEN_ONE, I32_RANGE.0, I32_RANGE.1, "fc0 bias")?);
        }
        for o in 0..l2 {
            for i in 0..l1 {
                let w = (l1w[i * STACKS * l2 + stack * l2 + o] + l1f[i * l2 + o])
                    .clamp(-127.0 / FC0_SCALE, 127.0 / FC0_SCALE);
                s.fc0_weights.push(quantise(w, FC0_SCALE, I8_RANGE.0, I8_RANGE.1, "fc0 weight")?);
            }
        }
        for o in 0..l3 {
            s.fc1_biases.push(quantise(
                l2b[stack * l3 + o],
                FC1_SCALE * HIDDEN_ONE,
                I32_RANGE.0,
                I32_RANGE.1,
                "fc1 bias",
            )?);
        }
        for o in 0..l3 {
            for i in 0..2 * l2 {
                s.fc1_weights.push(quantise(
                    l2w[i * STACKS * l3 + stack * l3 + o],
                    FC1_SCALE,
                    I8_RANGE.0,
                    I8_RANGE.1,
                    "fc1 weight",
                )?);
            }
        }
        s.fc2_bias = quantise(l3b[stack], FC2_SCALE * HIDDEN_ONE, I32_RANGE.0, I32_RANGE.1, "fc2 bias")?;
        for i in 0..2 * l2 + 2 * l3 {
            s.fc2_weights.push(quantise(l3w[i * STACKS + stack], FC2_SCALE, I8_RANGE.0, I8_RANGE.1, "fc2 weight")?);
        }
        stacks.push(s);
    }
    Ok(QuantNet { ft_biases, threat_weights, threat_psqt, halfka_weights, halfka_psqt, stacks })
}

/// Bullet weights of an integer net (factorisers zero; HalfKAv2_hm kings
/// split back into own/opponent planes as nnue-pytorch's
/// load_export_weights does).
fn dequantise_net(spec: Spec, q: &QuantNet) -> Weights {
    let Spec { l1, l2, l3, input_set, .. } = spec;
    let deq = |values: &[i64], scale: f32| values.iter().map(|&v| v as f32 / scale).collect::<Vec<f32>>();
    let mut weights = Weights::new();
    weights.insert("l0b".into(), deq(&q.ft_biases, FT_ONE));
    weights.insert("l0t".into(), deq(&q.threat_weights, FT_ONE));
    weights.insert("p0t".into(), deq(&q.threat_psqt, PSQT_SCALE));
    let mut l0h = vec![0.0; HALFKA_FEATURES * l1];
    let mut p0h = vec![0.0; HALFKA_FEATURES * PSQT_BUCKETS];
    for row in 0..input_set.export_rows() {
        let target = input_set.export_source_row(row);
        for o in 0..l1 {
            l0h[target * l1 + o] = q.halfka_weights[row * l1 + o] as f32 / FT_ONE;
        }
        for o in 0..PSQT_BUCKETS {
            p0h[target * PSQT_BUCKETS + o] = q.halfka_psqt[row * PSQT_BUCKETS + o] as f32 / PSQT_SCALE;
        }
    }
    weights.insert("l0h".into(), l0h);
    weights.insert("p0h".into(), p0h);
    weights.insert("l0v".into(), vec![0.0; PLANES * l1]);
    weights.insert("p0v".into(), vec![0.0; PLANES * PSQT_BUCKETS]);

    let mut l1w = vec![0.0; STACKS * l2 * l1];
    let mut l1b = vec![0.0; STACKS * l2];
    let mut l2w = vec![0.0; STACKS * l3 * 2 * l2];
    let mut l2b = vec![0.0; STACKS * l3];
    let mut l3w = vec![0.0; STACKS * (2 * l2 + 2 * l3)];
    let mut l3b = vec![0.0; STACKS];
    for (stack, s) in q.stacks.iter().enumerate() {
        for o in 0..l2 {
            l1b[stack * l2 + o] = s.fc0_biases[o] as f32 / (FC0_SCALE * HIDDEN_ONE);
            for i in 0..l1 {
                l1w[i * STACKS * l2 + stack * l2 + o] = s.fc0_weights[o * l1 + i] as f32 / FC0_SCALE;
            }
        }
        for o in 0..l3 {
            l2b[stack * l3 + o] = s.fc1_biases[o] as f32 / (FC1_SCALE * HIDDEN_ONE);
            for i in 0..2 * l2 {
                l2w[i * STACKS * l3 + stack * l3 + o] = s.fc1_weights[o * 2 * l2 + i] as f32 / FC1_SCALE;
            }
        }
        l3b[stack] = s.fc2_bias as f32 / (FC2_SCALE * HIDDEN_ONE);
        for i in 0..2 * l2 + 2 * l3 {
            l3w[i * STACKS + stack] = s.fc2_weights[i] as f32 / FC2_SCALE;
        }
    }
    weights.insert("l1w".into(), l1w);
    weights.insert("l1b".into(), l1b);
    weights.insert("l1f".into(), vec![0.0; l2 * l1]);
    weights.insert("l1fb".into(), vec![0.0; l2]);
    weights.insert("l2w".into(), l2w);
    weights.insert("l2b".into(), l2b);
    weights.insert("l3w".into(), l3w);
    weights.insert("l3b".into(), l3b);
    weights
}

fn push_i8s(out: &mut Vec<u8>, values: &[i64], padded: usize) {
    out.extend(values.iter().map(|&q| q as i8 as u8));
    out.resize(out.len() + padded - values.len(), 0);
}

fn push_i32(out: &mut Vec<u8>, value: i64) {
    out.extend_from_slice(&(value as i32).to_le_bytes());
}

/// Builds the .nnue bytes from Bullet weights (HalfKAv2_hm, piece-count
/// buckets only).
pub fn export_nnue(spec: Spec, weights: &Weights, description: &str) -> Result<Vec<u8>, String> {
    if spec.input_set != InputSet::HalfKav2Hm {
        return Err(format!("input_set {} has no Stockfish .nnue form; export net_format enyo-v1", spec.input_set.name()));
    }
    if spec.bucket_scheme != BucketScheme::PieceCount {
        return Err(format!(
            "bucket_scheme {} has no Stockfish .nnue form; export net_format enyo-v1",
            spec.bucket_scheme.name()
        ));
    }
    let q = quantise_net(spec, weights)?;
    let Spec { l1, l2, l3, .. } = spec;
    let mut out = Vec::new();
    let u32le = |out: &mut Vec<u8>, v: u32| out.extend_from_slice(&v.to_le_bytes());
    u32le(&mut out, VERSION);
    u32le(&mut out, spec.network_hash());
    u32le(&mut out, description.len() as u32);
    out.extend_from_slice(description.as_bytes());
    u32le(&mut out, spec.feature_hash() ^ (l1 as u32 * 2));

    encode_leb128(q.ft_biases.iter().copied(), &mut out);
    // Full_Threats: int8 rows, uncompressed, then LEB128 PSQT.
    push_i8s(&mut out, &q.threat_weights, q.threat_weights.len());
    encode_leb128(q.threat_psqt.iter().copied(), &mut out);
    encode_leb128(q.halfka_weights.iter().copied(), &mut out);
    encode_leb128(q.halfka_psqt.iter().copied(), &mut out);

    let fc_hash = spec.fc_hash();
    let pad32 = |n: usize| n.div_ceil(32) * 32;
    for s in &q.stacks {
        u32le(&mut out, fc_hash);
        s.fc0_biases.iter().for_each(|&b| push_i32(&mut out, b));
        for row in s.fc0_weights.chunks_exact(l1) {
            push_i8s(&mut out, row, pad32(l1));
        }
        s.fc1_biases.iter().for_each(|&b| push_i32(&mut out, b));
        for row in s.fc1_weights.chunks_exact(2 * l2) {
            push_i8s(&mut out, row, pad32(2 * l2));
        }
        push_i32(&mut out, s.fc2_bias);
        push_i8s(&mut out, &s.fc2_weights, pad32(2 * l2 + 2 * l3));
    }
    Ok(out)
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8], String> {
        let slice = self
            .bytes
            .get(self.offset..self.offset + n)
            .ok_or_else(|| format!("truncated net while reading {what} at {}", self.offset))?;
        self.offset += n;
        Ok(slice)
    }

    fn u32(&mut self, what: &str) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4, what)?.try_into().unwrap()))
    }

    fn expect_u32(&mut self, expected: u32, what: &str) -> Result<(), String> {
        let actual = self.u32(what)?;
        if actual != expected {
            return Err(format!("{what}: expected {expected:08x}, got {actual:08x}"));
        }
        Ok(())
    }

    /// Stockfish LEB128 block (magic, size, varints).
    fn leb128(&mut self, count: usize, what: &str) -> Result<Vec<i64>, String> {
        if self.take(LEB128_MAGIC.len(), what)? != LEB128_MAGIC {
            return Err(format!("{what}: missing LEB128 magic"));
        }
        self.leb128_sized(count, what)
    }

    /// Format-1 LEB128 block (size, varints).
    fn leb128_sized(&mut self, count: usize, what: &str) -> Result<Vec<i64>, String> {
        let len = self.u32(what)? as usize;
        let data = self.take(len, what)?;
        let mut values = Vec::with_capacity(count);
        let mut cursor = 0;
        for _ in 0..count {
            let mut value = 0_i64;
            let mut shift = 0;
            loop {
                let byte = *data.get(cursor).ok_or_else(|| format!("{what}: truncated LEB128 payload"))?;
                cursor += 1;
                value |= i64::from(byte & 0x7f) << shift;
                shift += 7;
                if byte & 0x80 == 0 {
                    if byte & 0x40 != 0 && shift < 64 {
                        value |= -1_i64 << shift;
                    }
                    break;
                }
            }
            values.push(value);
        }
        if cursor != data.len() {
            return Err(format!("{what}: {} unused LEB128 bytes", data.len() - cursor));
        }
        Ok(values)
    }

    fn i8s(&mut self, count: usize, what: &str) -> Result<Vec<i64>, String> {
        Ok(self.take(count, what)?.iter().map(|&b| i64::from(b as i8)).collect())
    }

    fn i16s(&mut self, count: usize, what: &str) -> Result<Vec<i64>, String> {
        Ok(self
            .take(2 * count, what)?
            .chunks_exact(2)
            .map(|c| i64::from(i16::from_le_bytes(c.try_into().unwrap())))
            .collect())
    }

    fn i32s(&mut self, count: usize, what: &str) -> Result<Vec<i64>, String> {
        Ok(self
            .take(4 * count, what)?
            .chunks_exact(4)
            .map(|c| i64::from(i32::from_le_bytes(c.try_into().unwrap())))
            .collect())
    }

    /// An i8 [rows][padded] block, the padding dropped.
    fn i8_rows(&mut self, rows: usize, cols: usize, padded: usize, what: &str) -> Result<Vec<i64>, String> {
        let raw = self.i8s(rows * padded, what)?;
        Ok(raw.chunks_exact(padded).flat_map(|row| row[..cols].iter().copied()).collect())
    }
}

/// Reads a Full_Threats+HalfKAv2_hm^ .nnue into Bullet weights (factorisers
/// zero, kings split back into own/opponent planes as nnue-pytorch's
/// load_export_weights does). Returns the weights and the description.
pub fn import_nnue(spec: Spec, bytes: &[u8]) -> Result<(Weights, String), String> {
    spec.validate()?;
    if spec.input_set != InputSet::HalfKav2Hm {
        return Err(format!("a Stockfish .nnue holds HalfKAv2_hm, not input_set {}", spec.input_set.name()));
    }
    if spec.bucket_scheme != BucketScheme::PieceCount {
        return Err(format!(
            "a Stockfish .nnue has piece-count buckets, not bucket_scheme {}",
            spec.bucket_scheme.name()
        ));
    }
    let Spec { l1, l2, l3, input_set, .. } = spec;
    let mut r = Reader { bytes, offset: 0 };
    r.expect_u32(VERSION, "version")?;
    r.expect_u32(spec.network_hash(), "network hash")?;
    let description_len = r.u32("description length")? as usize;
    let description = String::from_utf8_lossy(r.take(description_len, "description")?).into_owned();
    r.expect_u32(spec.feature_hash() ^ (l1 as u32 * 2), "feature transformer hash")?;

    let ft_biases = r.leb128(l1, "ft bias")?;
    let threat_weights = r.i8s(THREAT_FEATURES * l1, "threat weights")?;
    let threat_psqt = r.leb128(THREAT_FEATURES * PSQT_BUCKETS, "threat psqt")?;
    let halfka_weights = r.leb128(input_set.export_rows() * l1, "halfka weights")?;
    let halfka_psqt = r.leb128(input_set.export_rows() * PSQT_BUCKETS, "halfka psqt")?;

    let pad32 = |n: usize| n.div_ceil(32) * 32;
    let mut stacks = Vec::with_capacity(STACKS);
    for _ in 0..STACKS {
        r.expect_u32(spec.fc_hash(), "layer stack hash")?;
        stacks.push(QuantStack {
            fc0_biases: r.i32s(l2, "fc0 bias")?,
            fc0_weights: r.i8_rows(l2, l1, pad32(l1), "fc0 weights")?,
            fc1_biases: r.i32s(l3, "fc1 bias")?,
            fc1_weights: r.i8_rows(l3, 2 * l2, pad32(2 * l2), "fc1 weights")?,
            fc2_bias: r.i32s(1, "fc2 bias")?[0],
            fc2_weights: r.i8_rows(1, 2 * l2 + 2 * l3, pad32(2 * l2 + 2 * l3), "fc2 weights")?,
        });
    }
    if r.offset != bytes.len() {
        return Err(format!("{} trailing bytes after the last layer stack", bytes.len() - r.offset));
    }
    let q = QuantNet { ft_biases, threat_weights, threat_psqt, halfka_weights, halfka_psqt, stacks };
    Ok((dequantise_net(spec, &q), description))
}

// ---------------------------------------------------------------------------
// Enyo format-1 container, architecture 26 (tools/lib/enyo_net_v1.py and the
// engine's src/nnue/enyo/enyo_nn_loader.cpp)

pub const ENYO_MAGIC: &[u8; 8] = b"ENYONNUE";
const ENYO_FORMAT_VERSION: u32 = 1;
const ENYO_ARCHITECTURE: u32 = 26;
const ENYO_HEADER_SIZE: usize = 40;

fn fnv1a64(data: &[u8]) -> u64 {
    const PRIME: u64 = 0x100_0000_01b3;
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    let words = data.chunks_exact(8);
    let tail = words.remainder();
    for word in words {
        h = (h ^ u64::from_le_bytes(word.try_into().unwrap())).wrapping_mul(PRIME);
    }
    for &b in tail {
        h = (h ^ u64::from(b)).wrapping_mul(PRIME);
    }
    h
}

pub fn is_enyo_net(bytes: &[u8]) -> bool {
    bytes.starts_with(ENYO_MAGIC)
}

/// The architecture-26 layout block of `spec`.
fn enyo_layout(spec: Spec) -> [u32; 8] {
    [
        spec.input_set.layout_id(),
        1,
        spec.l1 as u32,
        PSQT_BUCKETS as u32,
        STACKS as u32,
        spec.l2 as u32,
        spec.l3 as u32,
        spec.bucket_scheme.layout_id(),
    ]
}

/// Builds an architecture-26 net (any input set and bucket scheme) from
/// Bullet weights. `metadata` is written as "key=value" lines in order (name,
/// trained). The engine loads only `Spec::engine_widths`.
pub fn export_enyo_v1(spec: Spec, weights: &Weights, metadata: &[(String, String)]) -> Result<Vec<u8>, String> {
    let q = quantise_net(spec, weights)?;
    let mut payload = Vec::new();
    for field in enyo_layout(spec) {
        payload.extend_from_slice(&field.to_le_bytes());
    }
    payload.extend(q.ft_biases.iter().flat_map(|&b| (b as i16).to_le_bytes()));
    encode_leb128_sized(q.halfka_weights.iter().copied(), &mut payload);
    encode_leb128_sized(q.halfka_psqt.iter().copied(), &mut payload);
    push_i8s(&mut payload, &q.threat_weights, q.threat_weights.len());
    encode_leb128_sized(q.threat_psqt.iter().copied(), &mut payload);
    for s in &q.stacks {
        s.fc0_biases.iter().for_each(|&b| push_i32(&mut payload, b));
        push_i8s(&mut payload, &s.fc0_weights, s.fc0_weights.len());
        s.fc1_biases.iter().for_each(|&b| push_i32(&mut payload, b));
        push_i8s(&mut payload, &s.fc1_weights, s.fc1_weights.len());
        push_i32(&mut payload, s.fc2_bias);
        push_i8s(&mut payload, &s.fc2_weights, s.fc2_weights.len());
    }

    let meta: String = metadata.iter().map(|(k, v)| format!("{k}={v}\n")).collect();
    let mut out = Vec::with_capacity(ENYO_HEADER_SIZE + meta.len() + payload.len());
    out.extend_from_slice(ENYO_MAGIC);
    out.extend_from_slice(&ENYO_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&ENYO_ARCHITECTURE.to_le_bytes());
    out.extend_from_slice(&(meta.len() as u32).to_le_bytes());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&fnv1a64(&payload).to_le_bytes());
    out.extend_from_slice(&[0; 8]);
    out.extend_from_slice(meta.as_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Reads an architecture-26 net into Bullet weights (factorisers zero).
/// Its input_set must be the one `spec` selects. Returns the weights and the
/// metadata lines.
pub fn import_enyo_v1(spec: Spec, bytes: &[u8]) -> Result<(Weights, Vec<(String, String)>), String> {
    spec.validate()?;
    let mut r = Reader { bytes, offset: 0 };
    if r.take(8, "magic")? != ENYO_MAGIC {
        return Err("not an Enyo format-1 net".to_string());
    }
    r.expect_u32(ENYO_FORMAT_VERSION, "format version")?;
    r.expect_u32(ENYO_ARCHITECTURE, "architecture")?;
    let meta_size = r.u32("metadata size")? as usize;
    let payload_size = r.u32("payload size")? as usize;
    let checksum = u64::from_le_bytes(r.take(8, "checksum")?.try_into().unwrap());
    if r.take(8, "reserved")?.iter().any(|&b| b != 0) {
        return Err("header reserved bytes are not zero".to_string());
    }
    let meta = String::from_utf8_lossy(r.take(meta_size, "metadata")?).into_owned();
    if bytes.len() != ENYO_HEADER_SIZE + meta_size + payload_size {
        return Err(format!("size mismatch: header declares {} bytes, file has {}", ENYO_HEADER_SIZE + meta_size + payload_size, bytes.len()));
    }
    if fnv1a64(&bytes[r.offset..]) != checksum {
        return Err("checksum mismatch".to_string());
    }
    let metadata = meta
        .lines()
        .filter_map(|line| line.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();

    let mut layout = [0_u32; 8];
    for field in &mut layout {
        *field = r.u32("layout")?;
    }
    if layout[0] != spec.input_set.layout_id() {
        return Err(format!("net has input_set {}, the architecture selects {} ({})", layout[0], spec.input_set.layout_id(), spec.input_set.name()));
    }
    if layout[7] != spec.bucket_scheme.layout_id() {
        return Err(format!(
            "net has bucket_scheme {}, the architecture selects {} ({})",
            layout[7],
            spec.bucket_scheme.layout_id(),
            spec.bucket_scheme.name()
        ));
    }
    if layout != enyo_layout(spec) {
        return Err(format!("net layout {layout:?} does not match the architecture ({:?})", enyo_layout(spec)));
    }
    let Spec { l1, l2, l3, input_set, .. } = spec;
    let ft_biases = r.i16s(l1, "ft biases")?;
    let halfka_weights = r.leb128_sized(input_set.export_rows() * l1, "halfka weights")?;
    let halfka_psqt = r.leb128_sized(input_set.export_rows() * PSQT_BUCKETS, "halfka psqt")?;
    let threat_weights = r.i8s(THREAT_FEATURES * l1, "threat weights")?;
    let threat_psqt = r.leb128_sized(THREAT_FEATURES * PSQT_BUCKETS, "threat psqt")?;
    let mut stacks = Vec::with_capacity(STACKS);
    for _ in 0..STACKS {
        stacks.push(QuantStack {
            fc0_biases: r.i32s(l2, "fc0 bias")?,
            fc0_weights: r.i8s(l2 * l1, "fc0 weights")?,
            fc1_biases: r.i32s(l3, "fc1 bias")?,
            fc1_weights: r.i8s(l3 * 2 * l2, "fc1 weights")?,
            fc2_bias: r.i32s(1, "fc2 bias")?[0],
            fc2_weights: r.i8s(2 * l2 + 2 * l3, "fc2 weights")?,
        });
    }
    if r.offset != bytes.len() {
        return Err(format!("{} trailing bytes after the last layer stack", bytes.len() - r.offset));
    }
    let q = QuantNet { ft_biases, threat_weights, threat_psqt, halfka_weights, halfka_psqt, stacks };
    Ok((dequantise_net(spec, &q), metadata))
}

/// (psqt_stm - psqt_nstm) / 2 of the position's bucket, in network units
/// (multiply by NNUE2SCORE for Stockfish internal units).
pub fn psqt_term(spec: Spec, weights: &Weights, pos: &ChessBoard) -> Result<f32, String> {
    let p0t = tensor(spec, weights, "p0t")?;
    let p0h = tensor(spec, weights, "p0h")?;
    let p0v = tensor(spec, weights, "p0v")?;
    let bucket = spec.bucket_scheme.bucket(pos);
    let row = |feature: usize| {
        if feature < THREAT_FEATURES {
            p0t[feature * PSQT_BUCKETS + bucket]
        } else {
            let f = feature - THREAT_FEATURES;
            p0h[f * PSQT_BUCKETS + bucket] + p0v[(f % PLANES) * PSQT_BUCKETS + bucket]
        }
    };
    let mut total = 0.0_f64;
    LayerStackInputs { input_set: spec.input_set }
        .map_features(pos, |stm, nstm| total += f64::from(row(stm)) - f64::from(row(nstm)));
    Ok((total / 2.0) as f32)
}

// ---------------------------------------------------------------------------
// Targets and data selection (nnue-pytorch defaults)

/// nnue-pytorch remap_tablebase_score: mate/TB scores (|s| >= 32000 - 245)
/// become base + decay^plies * scale.
pub fn remap_tablebase_score(score: i16, base: f32, scale: f32, decay: f32) -> i16 {
    const MATE: i32 = 32000;
    let abs = i32::from(score).abs();
    if abs < MATE - 245 {
        return score;
    }
    let plies = (MATE - abs).max(0);
    let remapped = (base + decay.powi(plies) * scale).round().clamp(0.0, f32::from(i16::MAX));
    let remapped = remapped as i16;
    if score < 0 { -remapped } else { remapped }
}

#[derive(Clone, Copy, Debug)]
pub struct TablebaseRemap {
    pub base: f32,
    pub scale: f32,
    pub decay: f32,
}

/// Applies the tablebase remap to every score a loader yields.
#[derive(Clone)]
pub struct RemappedLoader<L> {
    pub inner: L,
    pub remap: Option<TablebaseRemap>,
}

impl<L: DataLoader<ChessBoard>> DataLoader<ChessBoard> for RemappedLoader<L> {
    fn data_file_paths(&self) -> &[String] {
        self.inner.data_file_paths()
    }

    fn count_positions(&self) -> Option<u64> {
        self.inner.count_positions()
    }

    fn map_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, start_position: usize, mut f: F) {
        let Some(remap) = self.remap else {
            return self.inner.map_chunks(start_position, f);
        };
        // One reused buffer: a fresh copy of every (shuffle-buffer sized)
        // chunk costs an allocation and its page faults on the batch path.
        let mut out = Vec::new();
        self.inner.map_chunks(start_position, |chunk| {
            out.clear();
            out.extend(chunk.iter().map(|board| {
                let mut board = *board;
                board.score = remap_tablebase_score(board.score, remap.base, remap.scale, remap.decay);
                board
            }));
            f(&out)
        });
    }
}

/// nnue-pytorch's skip predicate (data_loader make_skip_predicate) with its
/// defaults: VALUE_NONE and placeholder-zero scores, early_fen_skipping,
/// random_fen_skipping, captures/checks, the WLD filter, the soft early-ply
/// filter (20 plies) and the piece-count distribution filter. Returns true
/// to keep the entry (Bullet filter convention).
#[derive(Clone, Copy, Debug)]
pub struct NptSkipConfig {
    /// Entries with ply <= early_fen_skipping are dropped (-1 disables).
    pub early_fen_skipping: i32,
    pub random_fen_skipping: u32,
    pub filtered: bool,
    pub wld_filtered: bool,
    pub soft_early_fen_skipping: i32,
    pub piece_count_filter: bool,
}

const PC_TARGET: [f64; 5] = [0.0, 0.4, 1.0, 1.0, 0.75];
const PLY_POINTS: [(f64, f64); 4] = [(0.0, 0.1), (6.0, 0.15), (10.0, 0.25), (18.0, 0.75)];
const MAX_PC_SKIP_RATE: f64 = 0.975;
/// Entries a thread counts before it folds its piece-count histogram into the
/// shared one and refreshes its acceptance table.
const PC_SYNC: u32 = 4096;
/// The shared histogram halves every this many entries (nnue-pytorch halves a
/// per-thread histogram every 10000 entries; the shared one sees ~threads x).
const PC_HALF_LIFE: f64 = 160_000.0;

fn pc_target_weight(pc: usize) -> f64 {
    let x = pc as f64;
    let y = PC_TARGET;
    if x <= 0.0 {
        return y[0];
    }
    if x >= 32.0 {
        return y[4];
    }
    let i = ((x / 8.0) as usize).min(3);
    let t = (x - i as f64 * 8.0) / 8.0;
    let slope = |idx: usize| match idx {
        0 => y[1] - y[0],
        4 => y[4] - y[3],
        _ => (y[idx + 1] - y[idx - 1]) / 2.0,
    };
    let (m0, m1) = (slope(i), slope(i + 1));
    let (t2, t3) = (t * t, t * t * t);
    let v = (2.0 * t3 - 3.0 * t2 + 1.0) * y[i]
        + (t3 - 2.0 * t2 + t) * m0
        + (-2.0 * t3 + 3.0 * t2) * y[i + 1]
        + (t3 - t2) * m1;
    v.max(0.0)
}

fn soft_ply_accept(ply: f64, end: f64) -> f64 {
    let mut points = PLY_POINTS.to_vec();
    points.push((end, 1.0));
    if ply <= points[0].0 {
        return points[0].1;
    }
    if ply >= points[4].0 {
        return points[4].1;
    }
    for w in points.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        if ply >= x0 && ply <= x1 {
            return if x1 == x0 { y0 } else { (y0 + (ply - x0) / (x1 - x0) * (y1 - y0)).clamp(0.0, 1.0) };
        }
    }
    1.0
}

/// Stockfish's win-rate model as used by nnue-pytorch's WLD filter.
fn score_result_prob(score: i16, ply: u16, result: i16) -> f64 {
    let m = f64::from(ply.min(240)) / 64.0;
    let a = (((-3.68389304 * m + 30.07065921) * m - 60.52878723) * m) + 149.53378557;
    let b = 1.5 * ((((-2.0181857 * m + 15.85685038) * m - 29.83452023) * m) + 47.59078827);
    let x = (100.0 * f64::from(score) / 208.0).clamp(-2000.0, 2000.0);
    let w = 1.0 / (1.0 + ((a - x) / b).exp());
    let l = 1.0 / (1.0 + ((a + x) / b).exp());
    match result.signum() {
        1 => w,
        -1 => l,
        _ => 1.0 - w - l,
    }
}

struct PieceCountHistogram {
    counts: [f64; 33],
    total: f64,
    since_halving: f64,
}

static PIECE_COUNTS: Mutex<PieceCountHistogram> =
    Mutex::new(PieceCountHistogram { counts: [0.0; 33], total: 0.0, since_halving: 0.0 });

thread_local! {
    static RNG_STATE: Cell<u64> = const { Cell::new(0) };
    static LAST_PLY: Cell<i32> = const { Cell::new(-1) };
    static LAST_SCORE: Cell<i32> = const { Cell::new(32002) };
    static PC_LOCAL: Cell<([u32; 33], u32)> = const { Cell::new(([0; 33], 0)) };
    static PC_ACCEPT: Cell<Option<[f64; 33]>> = const { Cell::new(None) };
}

fn uniform01() -> f64 {
    RNG_STATE.with(|state| {
        let mut x = state.get();
        if x == 0 {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            std::thread::current().id().hash(&mut hasher);
            std::time::SystemTime::now().hash(&mut hasher);
            x = hasher.finish() | 1;
        }
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state.set(x);
        (x >> 11) as f64 / (1_u64 << 53) as f64
    })
}

fn piece_count_accept(pc: usize) -> f64 {
    let (mut counts, mut n) = PC_LOCAL.with(Cell::get);
    counts[pc] += 1;
    n += 1;
    let mut accept = PC_ACCEPT.with(Cell::get);
    if accept.is_none() || n >= PC_SYNC {
        let mut shared = PIECE_COUNTS.lock().unwrap();
        for (total, local) in shared.counts.iter_mut().zip(counts) {
            *total += f64::from(local);
        }
        shared.total += f64::from(n);
        shared.since_halving += f64::from(n);
        if shared.since_halving >= PC_HALF_LIFE {
            shared.counts.iter_mut().for_each(|c| *c *= 0.5);
            shared.total *= 0.5;
            shared.since_halving = 0.0;
        }
        let weights: Vec<f64> = (0..33).map(pc_target_weight).collect();
        let weight_total: f64 = weights.iter().sum();
        let ratio = |i: usize| (shared.total * weights[i]) / (weight_total * shared.counts[i]);
        let min_ratio = (0..33)
            .filter(|&i| weights[i] > 0.0 && shared.counts[i] > 0.0)
            .map(ratio)
            .fold(f64::INFINITY, f64::min);
        let alpha = if min_ratio.is_finite() && min_ratio > 0.0 { (1.0 - MAX_PC_SKIP_RATE) / min_ratio } else { 1.0 };
        let mut table = [0.0; 33];
        for (i, slot) in table.iter_mut().enumerate() {
            if weights[i] > 0.0 && shared.counts[i] > 0.0 {
                *slot = (alpha * ratio(i)).clamp(0.0, 1.0);
            }
        }
        drop(shared);
        accept = Some(table);
        PC_ACCEPT.with(|a| a.set(accept));
        counts = [0; 33];
        n = 0;
    }
    PC_LOCAL.with(|c| c.set((counts, n)));
    accept.unwrap()[pc]
}

pub fn npt_skip_predicate(config: NptSkipConfig) -> impl Fn(&TrainingDataEntry) -> bool + Clone + Send + Sync {
    const VALUE_NONE: i32 = 32002;
    let skip_prob = if config.random_fen_skipping > 0 {
        f64::from(config.random_fen_skipping) / f64::from(config.random_fen_skipping + 1)
    } else {
        0.0
    };
    move |e: &TrainingDataEntry| {
        let score = i32::from(e.score);
        let ply = i32::from(e.ply);
        let last_ply = LAST_PLY.with(Cell::get);
        let last_score = LAST_SCORE.with(Cell::get);
        let placeholder_zero =
            ply > last_ply && last_score != VALUE_NONE && last_score.abs() > 100 && e.result != 0 && score == 0;
        LAST_PLY.with(|c| c.set(ply));
        if score == VALUE_NONE || placeholder_zero {
            return false;
        }
        LAST_SCORE.with(|c| c.set(score));
        if ply <= config.early_fen_skipping {
            return false;
        }
        if skip_prob > 0.0 && uniform01() < skip_prob {
            return false;
        }
        if config.filtered {
            let to = e.pos.piece_at(e.mv.to());
            let capture = to.piece_type() != PieceType::None && to.color() != e.pos.piece_at(e.mv.from()).color();
            if capture || e.pos.is_checked(e.pos.side_to_move()) {
                return false;
            }
        }
        if config.wld_filtered && uniform01() < 1.0 - score_result_prob(e.score, e.ply, e.result) {
            return false;
        }
        if config.soft_early_fen_skipping > 0
            && ply < config.soft_early_fen_skipping
            && uniform01() >= soft_ply_accept(f64::from(ply), f64::from(config.soft_early_fen_skipping))
        {
            return false;
        }
        if config.piece_count_filter {
            let pc = e.pos.occupied().count() as usize;
            if pc > 32 || uniform01() >= piece_count_accept(pc) {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn king_buckets_match_nnue_pytorch_table() {
        // nnue-pytorch KingBuckets rows a1..h1 and a8..h8 (files e-h).
        assert_eq!([4, 5, 6, 7].map(king_bucket), [31, 30, 29, 28]);
        assert_eq!([60, 61, 62, 63].map(king_bucket), [3, 2, 1, 0]);
        for bucket in 0..KING_BUCKETS {
            assert_eq!(king_bucket(bucket_king_square(bucket)), bucket);
        }
    }

    #[test]
    fn halfka_index_matches_nnue_pytorch() {
        // White king e1, white pawn e2, white point of view: no mirror,
        // bucket 31, plane 0.
        assert_eq!(halfka_index(0, true, 12, 4), 12 + 31 * PLANES);
        // King on d1 mirrors files: d1 -> e1, pawn e2 -> d2.
        assert_eq!(halfka_index(0, false, 12, 3), 11 + 64 + 31 * PLANES);
    }

    /// Architecture-24/25 FeatureIdx as the trainer computed it for those nets
    /// (main.rs enyo_feature::<32, 12>): `sq` and `king` a8 = 0 from white's
    /// side of the perspective's board, `view` 0 for the side to move.
    fn arch25_feature(piece: usize, sq: usize, king: usize, view: usize) -> usize {
        let colour = usize::from(piece & 8 != 0);
        let piece_code = ((piece & 7) << 1) | colour;
        let orient = (7 * usize::from(king & 4 == 0)) ^ (56 * view);
        let plane = 6 * ((piece_code ^ view) & 1) + (piece_code >> 1);
        ENYO_KING_BUCKETS[orient ^ king] * PLANES + plane * 64 + (orient ^ sq)
    }

    #[test]
    fn enyo_halfka_matches_the_architecture_25_inputs() {
        // As EnyoInputs::map_features fed arch25_feature: side to move with
        // our_ksq ^ 56, the other side with opp_ksq (already flipped).
        for piece in (0..6).chain(8..14) {
            let piece_type = piece & 7;
            let ours = piece & 8 == 0;
            for square in 0..64 {
                for king in 0..64 {
                    assert_eq!(
                        enyo_halfka_index(piece_type, ours, square, king),
                        arch25_feature(piece, square ^ 56, king ^ 56, 0)
                    );
                    assert_eq!(
                        enyo_halfka_index(piece_type, !ours, square ^ 56, king),
                        arch25_feature(piece, square ^ 56, king, 1)
                    );
                }
            }
        }
    }

    #[test]
    fn enyo_halfka_worked_examples() {
        // The engine's tests (enyo_halfka.worked_examples) in Bullet squares.
        assert_eq!(enyo_halfka_index(0, true, 12, 4), 52); // pawn e2, king e1
        assert_eq!(enyo_halfka_index(0, false, 52, 4), 6 * 64 + 12); // their pawn e7
        assert_eq!(enyo_halfka_index(0, true, 12, 3), 51); // king d1 mirrors e2 -> d2
        assert_eq!(enyo_halfka_index(5, true, 22, 22), 10 * PLANES + 5 * 64 + 46); // king g3
        assert_eq!(enyo_halfka_index(5, false, 62, 22), 10 * PLANES + 11 * 64 + 6); // their king g8
        let mut seen = vec![false; HALFKA_FEATURES];
        for piece_type in 0..6 {
            for own in [true, false] {
                for square in 0..64 {
                    for king in 0..64 {
                        seen[enyo_halfka_index(piece_type, own, square, king)] = true;
                    }
                }
            }
        }
        assert!(seen.iter().all(|&s| s), "every Enyo HalfKA row is reachable");
    }

    fn noisy_weights(spec: Spec) -> Weights {
        let mut weights = initial_weights(spec, 7);
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        for v in weights.get_mut("l0v").unwrap() {
            *v = (rng.random_range(-20..20) as f32) / FT_ONE;
        }
        for v in weights.get_mut("l1f").unwrap() {
            *v = (rng.random_range(-20..20) as f32) / FC0_SCALE;
        }
        weights
    }

    fn test_metadata() -> Vec<(String, String)> {
        vec![("name".into(), "test".into()), ("trained".into(), "2026-11-22T21:00Z".into())]
    }

    #[test]
    fn enyo_v1_export_import_export_is_identical() {
        for input_set in [InputSet::HalfKav2Hm, InputSet::EnyoHalfKa32x12] {
            let spec = Spec { l1: 64, l2: 4, l3: 4, input_set, bucket_scheme: BucketScheme::PieceCount };
            let first = export_enyo_v1(spec, &noisy_weights(spec), &test_metadata()).unwrap();
            assert!(is_enyo_net(&first));
            let layout = ENYO_HEADER_SIZE + u32::from_le_bytes(first[16..20].try_into().unwrap()) as usize;
            assert_eq!(u32::from_le_bytes(first[layout..layout + 4].try_into().unwrap()), input_set.layout_id());
            let (imported, metadata) = import_enyo_v1(spec, &first).unwrap();
            assert_eq!(metadata, test_metadata());
            assert_eq!(export_enyo_v1(spec, &imported, &metadata).unwrap(), first, "{input_set:?}");
            let other = Spec { input_set: if input_set == InputSet::HalfKav2Hm { InputSet::EnyoHalfKa32x12 } else { InputSet::HalfKav2Hm }, ..spec };
            assert!(import_enyo_v1(other, &first).unwrap_err().contains("input_set"));
        }
    }

    #[test]
    fn enyo_v1_layout_carries_fc0_out_and_bucket_scheme() {
        for (l2, bucket_scheme) in [(4, BucketScheme::Phase), (8, BucketScheme::PieceCount), (8, BucketScheme::Phase)] {
            for input_set in [InputSet::HalfKav2Hm, InputSet::EnyoHalfKa32x12] {
                let spec = Spec { l1: 64, l2, l3: 4, input_set, bucket_scheme };
                let first = export_enyo_v1(spec, &noisy_weights(spec), &test_metadata()).unwrap();
                let layout = ENYO_HEADER_SIZE + u32::from_le_bytes(first[16..20].try_into().unwrap()) as usize;
                let field = |i: usize| u32::from_le_bytes(first[layout + 4 * i..layout + 4 * i + 4].try_into().unwrap());
                assert_eq!((field(5), field(6), field(7)), (l2 as u32, 4, bucket_scheme.layout_id()));
                let (imported, metadata) = import_enyo_v1(spec, &first).unwrap();
                assert_eq!(export_enyo_v1(spec, &imported, &metadata).unwrap(), first, "{spec:?}");
                let other_scheme = match bucket_scheme {
                    BucketScheme::PieceCount => BucketScheme::Phase,
                    BucketScheme::Phase => BucketScheme::PieceCount,
                };
                let err = import_enyo_v1(Spec { bucket_scheme: other_scheme, ..spec }, &first).unwrap_err();
                assert!(err.contains("bucket_scheme"), "{err}");
                let err = import_enyo_v1(Spec { l2: l2 * 2, ..spec }, &first).unwrap_err();
                assert!(err.contains("does not match"), "{err}");
            }
        }
        let phase = Spec { l1: 64, l2: 4, l3: 4, bucket_scheme: BucketScheme::Phase, ..Spec::STOCKFISH };
        assert!(export_nnue(phase, &initial_weights(phase, 1), "test").unwrap_err().contains("bucket_scheme"));
    }

    #[test]
    fn engine_loadable_specs_and_their_published_form() {
        let wide = Spec { l2: 64, ..Spec::STOCKFISH };
        let phase = Spec { bucket_scheme: BucketScheme::Phase, ..Spec::STOCKFISH };
        let enyo = Spec { input_set: InputSet::EnyoHalfKa32x12, ..Spec::STOCKFISH };
        for spec in [Spec::STOCKFISH, wide, phase, Spec { l2: 64, ..phase }, enyo, Spec { l2: 64, ..enyo }] {
            assert!(spec.engine_widths(), "{spec:?}");
            assert!(Spec { l1: 1280, ..spec }.engine_widths(), "{spec:?}");
            assert!(!Spec { l1: 1280, ..spec }.nnue_form(), "{spec:?}");
        }
        assert!(!Spec { l2: 16, ..Spec::STOCKFISH }.engine_widths());
        assert!(!Spec { l1: 2048, ..Spec::STOCKFISH }.engine_widths());
        assert!(Spec::STOCKFISH.nnue_form());
        assert!(Spec { l1: 512, l2: 16, ..Spec::STOCKFISH }.nnue_form());
        assert!(Spec { l1: 2048, ..Spec::STOCKFISH }.nnue_form());
        assert!(!wide.nnue_form() && !phase.nnue_form() && !enyo.nnue_form());
    }

    fn board(fen: &str) -> ChessBoard {
        format!("{fen} | 0 | 0.5").parse::<ChessBoard>().unwrap()
    }

    #[test]
    fn bucket_schemes_match_the_engine() {
        // The engine's layer_stack_bucket.piece_count_and_phase cases.
        let buckets = |fen: &str| {
            let pos = board(fen);
            (BucketScheme::PieceCount.bucket(&pos), BucketScheme::Phase.bucket(&pos))
        };
        assert_eq!(buckets("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"), (7, 7));
        assert_eq!(buckets("8/2p5/3p4/1P1Pp1k1/4P3/5K2/8/8 w - - 0 40"), (1, 0));
        assert_eq!(buckets("3rk3/pppq1ppp/8/8/8/8/PPPQ1PPP/3RK3 w - - 0 1"), (4, 3));
        assert_eq!(buckets("4k3/8/8/8/8/8/8/3NK3 w - - 0 1").1, 0);
        assert_eq!(buckets("4k3/8/8/8/8/8/8/2BNK1N1 w - - 0 1").1, 0);
        assert_eq!(buckets("4k3/8/8/8/8/8/8/2RNK1N1 w - - 0 1").1, 1);
        assert_eq!(buckets("qqbqkbnr/pp1ppppp/8/8/8/8/PP1PPPPP/QQBQKBNR w Kk - 0 1").1, 7);
        assert_eq!(buckets("r1bqkb1r/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1").1, 6);
        assert_eq!(buckets("r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1").1, 7);
        // Side to move does not matter.
        assert_eq!(buckets("r1bqkb1r/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR b KQkq - 0 1").1, 6);
        // Every phase 0..=24 maps onto all eight buckets, three points each
        // above bucket 0.
        let by_phase: Vec<u32> = (0..=24).map(|phase: u32| (phase * 8 / 25).min(7)).collect();
        assert_eq!(by_phase[..4], [0; 4]);
        assert_eq!(by_phase[22..], [7; 3]);
    }

    #[test]
    fn enyo_v1_and_nnue_hold_the_same_halfkav2_hm_net() {
        let spec = Spec { l1: 64, l2: 4, l3: 4, ..Spec::STOCKFISH };
        let weights = noisy_weights(spec);
        let (from_nnue, _) = import_nnue(spec, &export_nnue(spec, &weights, "test").unwrap()).unwrap();
        let (from_enyo, _) = import_enyo_v1(spec, &export_enyo_v1(spec, &weights, &test_metadata()).unwrap()).unwrap();
        assert_eq!(from_nnue, from_enyo);
        let enyo = Spec { input_set: InputSet::EnyoHalfKa32x12, ..spec };
        assert!(export_nnue(enyo, &initial_weights(enyo, 1), "test").is_err());
    }

    #[test]
    fn hashes_match_the_engine() {
        // Values of the engine's FeatureTransformerHash / NetworkHash for
        // FullThreats + HalfKAv2_hm at L1 = 1024.
        let spec = Spec::STOCKFISH;
        assert_eq!(spec.feature_hash() ^ 2048, (FULL_THREATS_HASH.rotate_left(1) ^ HALFKA_HASH) ^ 2048);
        let mut hash = 0xEC42_E90D_u32 ^ 2048;
        let affine = |prev: u32, out: u32| (0xCC03_DAE4_u32.wrapping_add(out)) ^ (prev >> 1) ^ (prev << 31);
        hash = affine(hash, 32).wrapping_add(0x538D_24C7);
        hash = affine(hash, 32).wrapping_add(0x538D_24C7);
        hash = affine(hash, 1);
        assert_eq!(spec.fc_hash(), hash);
    }

    #[test]
    fn leb128_round_trips() {
        let values = vec![0, 1, -1, 63, 64, -64, -65, 127, 128, -32768, 32767, i32::MAX as i64, -(i32::MAX as i64)];
        let mut bytes = Vec::new();
        encode_leb128(values.clone(), &mut bytes);
        let mut r = Reader { bytes: &bytes, offset: 0 };
        assert_eq!(r.leb128(values.len(), "test").unwrap(), values);
    }

    #[test]
    fn export_import_export_is_identical() {
        let spec = Spec { l1: 64, l2: 4, l3: 4, ..Spec::STOCKFISH };
        let mut weights = initial_weights(spec, 7);
        // Non-zero factors exercise coalescing; import zeroes them, so
        // the second export must still match byte for byte.
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        for v in weights.get_mut("l0v").unwrap() {
            *v = (rng.random_range(-20..20) as f32) / FT_ONE;
        }
        for v in weights.get_mut("l1f").unwrap() {
            *v = (rng.random_range(-20..20) as f32) / FC0_SCALE;
        }
        let first = export_nnue(spec, &weights, "test").unwrap();
        let (imported, description) = import_nnue(spec, &first).unwrap();
        assert_eq!(description, "test");
        assert_eq!(export_nnue(spec, &imported, &description).unwrap(), first);
    }

    #[test]
    fn tablebase_remap_matches_nnue_pytorch() {
        assert_eq!(remap_tablebase_score(500, 4000.0, 4000.0, 0.85), 500);
        assert_eq!(remap_tablebase_score(32000, 4000.0, 4000.0, 0.85), 8000);
        assert_eq!(remap_tablebase_score(-31999, 4000.0, 4000.0, 0.85), -7400);
    }

    #[test]
    fn soft_early_ply_interpolates() {
        assert!((soft_ply_accept(19.0, 20.0) - 0.875).abs() < 1e-12);
        assert!((soft_ply_accept(3.0, 20.0) - 0.125).abs() < 1e-12);
    }
}
