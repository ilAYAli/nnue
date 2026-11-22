//! The nnue-pytorch "layer stacks" network (features Full_Threats+HalfKAv2_hm^)
//! as trained by nnue-pytorch at ccc0838^ and evaluated by the engine's
//! Stockfish `current` feature variant:
//!
//! * feature transformer: Full_Threats rows (int8 export) followed by
//!   HalfKAv2_hm^ rows (12 piece planes while training, one shared 768-row
//!   virtual factor; 11 planes with merged kings in the export), L1 outputs
//!   plus 8 PSQT outputs per row;
//! * per perspective clamp(acc, 0, 255/256), pairwise product of the two
//!   halves (L1/2 values), side to move first;
//! * 8 layer stacks picked by (pieces - 1) / 4: fc0 L1 -> L2 (factorised
//!   across stacks), [sqr(fc0), fc0] clipped to 127/128, fc1 2*L2 -> L3,
//!   [sqr(fc1), fc1] clipped, fc2 over [fc0 branch, fc1 branch] -> 1, plus
//!   fc0[L2-2] - fc0[L2-1] passthrough, plus (psqt_stm - psqt_nstm) / 2.
//!
//! Everything here is CPU-side and graph independent: input indexing, the
//! nnue-pytorch initialisation, Stockfish .nnue export/import (nnue-pytorch
//! serialize.py byte layout), the nnue-pytorch data skip predicate and the
//! tablebase score remap. The Bullet graph lives in main.rs.

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

/// Layer widths. The engine's Stockfish path runs 1024/32/32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub l1: usize,
    pub l2: usize,
    pub l3: usize,
}

impl Spec {
    pub const STOCKFISH: Spec = Spec { l1: 1024, l2: 32, l3: 32 };

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
        let Spec { l1, l2, l3 } = self;
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

#[derive(Clone, Copy, Debug, Default)]
pub struct LayerStackInputs;

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
        for (piece, square) in pos.into_iter() {
            let piece_type = usize::from(piece & 7);
            let ours = piece & 8 == 0;
            let square = usize::from(square);
            f(
                THREAT_FEATURES + halfka_index(piece_type, ours, square, stm_king),
                THREAT_FEATURES + halfka_index(piece_type, !ours, square ^ 56, nstm_king),
            );
        }
    }

    fn shorthand(&self) -> String {
        "full-threats+halfkav2hm".to_string()
    }

    fn description(&self) -> String {
        "nnue-pytorch Full_Threats+HalfKAv2_hm^ (12-plane training layout)".to_string()
    }
}

/// Layer-stack and PSQT bucket: (pieces - 1) / 4.
pub fn stack_index(pieces: u32) -> usize {
    ((pieces.max(1) - 1) / 4).min(STACKS as u32 - 1) as usize
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PieceCountBuckets;

impl OutputBuckets<ChessBoard> for PieceCountBuckets {
    const BUCKETS: usize = STACKS;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        stack_index(pos.occ().count_ones()) as u8
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
    let Spec { l1, l2, l3 } = spec;
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
        let plane = (feature % PLANES) / 64;
        let piece_type = plane / 2;
        if piece_type < 5 {
            let sign = if plane % 2 == 0 { 1.0 } else { -1.0 };
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

pub fn encode_leb128(values: impl IntoIterator<Item = i64>, out: &mut Vec<u8>) {
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
    out.extend_from_slice(LEB128_MAGIC);
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

/// Builds the .nnue bytes from Bullet weights.
pub fn export_nnue(spec: Spec, weights: &Weights, description: &str) -> Result<Vec<u8>, String> {
    spec.validate()?;
    let Spec { l1, l2, l3 } = spec;
    let mut out = Vec::new();
    let u32le = |out: &mut Vec<u8>, v: u32| out.extend_from_slice(&v.to_le_bytes());
    u32le(&mut out, VERSION);
    u32le(&mut out, spec.network_hash());
    u32le(&mut out, description.len() as u32);
    out.extend_from_slice(description.as_bytes());
    u32le(&mut out, spec.feature_hash() ^ (l1 as u32 * 2));

    let i16_range = (-(i16::MAX as i64), i16::MAX as i64);
    let i32_range = (-(i32::MAX as i64), i32::MAX as i64);
    let i8_range = (-(i8::MAX as i64), i8::MAX as i64);

    let l0b = tensor(spec, weights, "l0b")?;
    let biases = l0b
        .iter()
        .map(|&b| quantise(b, FT_ONE, i16_range.0, i16_range.1, "ft bias"))
        .collect::<Result<Vec<_>, _>>()?;
    encode_leb128(biases, &mut out);

    // Full_Threats: int8 rows, uncompressed, then LEB128 PSQT.
    let l0t = tensor(spec, weights, "l0t")?;
    out.reserve(l1 * THREAT_FEATURES);
    for &w in l0t {
        out.push(quantise(w, FT_ONE, i8_range.0, i8_range.1, "threat weight")? as i8 as u8);
    }
    let p0t = tensor(spec, weights, "p0t")?;
    let psqt = p0t
        .iter()
        .map(|&w| quantise(w, PSQT_SCALE, i32_range.0, i32_range.1, "threat psqt"))
        .collect::<Result<Vec<_>, _>>()?;
    encode_leb128(psqt, &mut out);

    // HalfKAv2_hm: coalesce the factor, merge kings, int16 + int32 PSQT.
    let l0h = tensor(spec, weights, "l0h")?;
    let l0v = tensor(spec, weights, "l0v")?;
    let p0h = tensor(spec, weights, "p0h")?;
    let p0v = tensor(spec, weights, "p0v")?;
    let mut halfka = Vec::with_capacity(HALFKA_EXPORT_FEATURES * l1);
    let mut halfka_psqt = Vec::with_capacity(HALFKA_EXPORT_FEATURES * PSQT_BUCKETS);
    for row in 0..HALFKA_EXPORT_FEATURES {
        let source = export_halfka_row(row);
        let plane = source % PLANES;
        for o in 0..l1 {
            let w = l0h[source * l1 + o] + l0v[plane * l1 + o];
            halfka.push(quantise(w, FT_ONE, i16_range.0, i16_range.1, "halfka weight")?);
        }
        for o in 0..PSQT_BUCKETS {
            let w = p0h[source * PSQT_BUCKETS + o] + p0v[plane * PSQT_BUCKETS + o];
            halfka_psqt.push(quantise(w, PSQT_SCALE, i32_range.0, i32_range.1, "halfka psqt")?);
        }
    }
    encode_leb128(halfka, &mut out);
    encode_leb128(halfka_psqt, &mut out);

    let l1w = tensor(spec, weights, "l1w")?;
    let l1b = tensor(spec, weights, "l1b")?;
    let l1f = tensor(spec, weights, "l1f")?;
    let l1fb = tensor(spec, weights, "l1fb")?;
    let l2w = tensor(spec, weights, "l2w")?;
    let l2b = tensor(spec, weights, "l2b")?;
    let l3w = tensor(spec, weights, "l3w")?;
    let l3b = tensor(spec, weights, "l3b")?;
    let fc_hash = spec.fc_hash();
    let pad32 = |n: usize| n.div_ceil(32) * 32;
    for stack in 0..STACKS {
        u32le(&mut out, fc_hash);
        // fc0: the graph clamps the merged weight to the int8 grid range.
        for o in 0..l2 {
            let b = l1b[stack * l2 + o] + l1fb[o];
            let q = quantise(b, FC0_SCALE * HIDDEN_ONE, i32_range.0, i32_range.1, "fc0 bias")?;
            out.extend_from_slice(&(q as i32).to_le_bytes());
        }
        for o in 0..l2 {
            for i in 0..pad32(l1) {
                let q = if i < l1 {
                    let w = (l1w[i * STACKS * l2 + stack * l2 + o] + l1f[i * l2 + o])
                        .clamp(-127.0 / FC0_SCALE, 127.0 / FC0_SCALE);
                    quantise(w, FC0_SCALE, i8_range.0, i8_range.1, "fc0 weight")?
                } else {
                    0
                };
                out.push(q as i8 as u8);
            }
        }
        for o in 0..l3 {
            let q = quantise(l2b[stack * l3 + o], FC1_SCALE * HIDDEN_ONE, i32_range.0, i32_range.1, "fc1 bias")?;
            out.extend_from_slice(&(q as i32).to_le_bytes());
        }
        for o in 0..l3 {
            for i in 0..pad32(2 * l2) {
                let q = if i < 2 * l2 {
                    quantise(l2w[i * STACKS * l3 + stack * l3 + o], FC1_SCALE, i8_range.0, i8_range.1, "fc1 weight")?
                } else {
                    0
                };
                out.push(q as i8 as u8);
            }
        }
        let q = quantise(l3b[stack], FC2_SCALE * HIDDEN_ONE, i32_range.0, i32_range.1, "fc2 bias")?;
        out.extend_from_slice(&(q as i32).to_le_bytes());
        for i in 0..pad32(2 * l2 + 2 * l3) {
            let q = if i < 2 * l2 + 2 * l3 {
                quantise(l3w[i * STACKS + stack], FC2_SCALE, i8_range.0, i8_range.1, "fc2 weight")?
            } else {
                0
            };
            out.push(q as i8 as u8);
        }
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
            .ok_or_else(|| format!("truncated .nnue while reading {what} at {}", self.offset))?;
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

    fn leb128(&mut self, count: usize, what: &str) -> Result<Vec<i64>, String> {
        if self.take(LEB128_MAGIC.len(), what)? != LEB128_MAGIC {
            return Err(format!("{what}: missing LEB128 magic"));
        }
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

    fn i32s(&mut self, count: usize, what: &str) -> Result<Vec<i64>, String> {
        Ok(self
            .take(4 * count, what)?
            .chunks_exact(4)
            .map(|c| i64::from(i32::from_le_bytes(c.try_into().unwrap())))
            .collect())
    }
}

/// Reads a Full_Threats+HalfKAv2_hm^ .nnue into Bullet weights (factorisers
/// zero, kings split back into own/opponent planes as nnue-pytorch's
/// load_export_weights does). Returns the weights and the description.
pub fn import_nnue(spec: Spec, bytes: &[u8]) -> Result<(Weights, String), String> {
    spec.validate()?;
    let Spec { l1, l2, l3 } = spec;
    let mut r = Reader { bytes, offset: 0 };
    r.expect_u32(VERSION, "version")?;
    r.expect_u32(spec.network_hash(), "network hash")?;
    let description_len = r.u32("description length")? as usize;
    let description = String::from_utf8_lossy(r.take(description_len, "description")?).into_owned();
    r.expect_u32(spec.feature_hash() ^ (l1 as u32 * 2), "feature transformer hash")?;

    let mut weights = Weights::new();
    let deq = |values: Vec<i64>, scale: f32| values.into_iter().map(|q| q as f32 / scale).collect::<Vec<f32>>();
    weights.insert("l0b".into(), deq(r.leb128(l1, "ft bias")?, FT_ONE));
    weights.insert("l0t".into(), deq(r.i8s(THREAT_FEATURES * l1, "threat weights")?, FT_ONE));
    weights.insert("p0t".into(), deq(r.leb128(THREAT_FEATURES * PSQT_BUCKETS, "threat psqt")?, PSQT_SCALE));
    let halfka = r.leb128(HALFKA_EXPORT_FEATURES * l1, "halfka weights")?;
    let halfka_psqt = r.leb128(HALFKA_EXPORT_FEATURES * PSQT_BUCKETS, "halfka psqt")?;
    let mut l0h = vec![0.0; HALFKA_FEATURES * l1];
    let mut p0h = vec![0.0; HALFKA_FEATURES * PSQT_BUCKETS];
    for row in 0..HALFKA_EXPORT_FEATURES {
        let target = export_halfka_row(row);
        for o in 0..l1 {
            l0h[target * l1 + o] = halfka[row * l1 + o] as f32 / FT_ONE;
        }
        for o in 0..PSQT_BUCKETS {
            p0h[target * PSQT_BUCKETS + o] = halfka_psqt[row * PSQT_BUCKETS + o] as f32 / PSQT_SCALE;
        }
    }
    weights.insert("l0h".into(), l0h);
    weights.insert("p0h".into(), p0h);
    weights.insert("l0v".into(), vec![0.0; PLANES * l1]);
    weights.insert("p0v".into(), vec![0.0; PLANES * PSQT_BUCKETS]);

    let pad32 = |n: usize| n.div_ceil(32) * 32;
    let mut l1w = vec![0.0; STACKS * l2 * l1];
    let mut l1b = vec![0.0; STACKS * l2];
    let mut l2w = vec![0.0; STACKS * l3 * 2 * l2];
    let mut l2b = vec![0.0; STACKS * l3];
    let mut l3w = vec![0.0; STACKS * (2 * l2 + 2 * l3)];
    let mut l3b = vec![0.0; STACKS];
    for stack in 0..STACKS {
        r.expect_u32(spec.fc_hash(), "layer stack hash")?;
        for (o, q) in r.i32s(l2, "fc0 bias")?.into_iter().enumerate() {
            l1b[stack * l2 + o] = q as f32 / (FC0_SCALE * HIDDEN_ONE);
        }
        let w = r.i8s(l2 * pad32(l1), "fc0 weights")?;
        for o in 0..l2 {
            for i in 0..l1 {
                l1w[i * STACKS * l2 + stack * l2 + o] = w[o * pad32(l1) + i] as f32 / FC0_SCALE;
            }
        }
        for (o, q) in r.i32s(l3, "fc1 bias")?.into_iter().enumerate() {
            l2b[stack * l3 + o] = q as f32 / (FC1_SCALE * HIDDEN_ONE);
        }
        let w = r.i8s(l3 * pad32(2 * l2), "fc1 weights")?;
        for o in 0..l3 {
            for i in 0..2 * l2 {
                l2w[i * STACKS * l3 + stack * l3 + o] = w[o * pad32(2 * l2) + i] as f32 / FC1_SCALE;
            }
        }
        l3b[stack] = r.i32s(1, "fc2 bias")?[0] as f32 / (FC2_SCALE * HIDDEN_ONE);
        let w = r.i8s(pad32(2 * l2 + 2 * l3), "fc2 weights")?;
        for i in 0..2 * l2 + 2 * l3 {
            l3w[i * STACKS + stack] = w[i] as f32 / FC2_SCALE;
        }
    }
    if r.offset != bytes.len() {
        return Err(format!("{} trailing bytes after the last layer stack", bytes.len() - r.offset));
    }
    weights.insert("l1w".into(), l1w);
    weights.insert("l1b".into(), l1b);
    weights.insert("l1f".into(), vec![0.0; l2 * l1]);
    weights.insert("l1fb".into(), vec![0.0; l2]);
    weights.insert("l2w".into(), l2w);
    weights.insert("l2b".into(), l2b);
    weights.insert("l3w".into(), l3w);
    weights.insert("l3b".into(), l3b);
    Ok((weights, description))
}

/// (psqt_stm - psqt_nstm) / 2 of the position's bucket, in network units
/// (multiply by NNUE2SCORE for Stockfish internal units).
pub fn psqt_term(spec: Spec, weights: &Weights, pos: &ChessBoard) -> Result<f32, String> {
    let p0t = tensor(spec, weights, "p0t")?;
    let p0h = tensor(spec, weights, "p0h")?;
    let p0v = tensor(spec, weights, "p0v")?;
    let bucket = stack_index(pos.occ().count_ones());
    let row = |feature: usize| {
        if feature < THREAT_FEATURES {
            p0t[feature * PSQT_BUCKETS + bucket]
        } else {
            let f = feature - THREAT_FEATURES;
            p0h[f * PSQT_BUCKETS + bucket] + p0v[(f % PLANES) * PSQT_BUCKETS + bucket]
        }
    };
    let mut total = 0.0_f64;
    LayerStackInputs.map_features(pos, |stm, nstm| total += f64::from(row(stm)) - f64::from(row(nstm)));
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
        let spec = Spec { l1: 64, l2: 4, l3: 4 };
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
