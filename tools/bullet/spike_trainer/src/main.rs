mod enyo_threats;

use std::{
    env,
    fs::File,
    io::{self, Write},
    path::Path,
};

use bullet_lib::{
    game::{
        formats::bulletformat::ChessBoard,
        inputs::{ChessBucketsMirrored, SparseInputType},
        outputs::{MaterialCount, OutputBuckets},
    },
    nn::{
        optimiser::{AdamW, AdamWParams},
        Affine, InitSettings, ModelBuilder, Shape,
    },
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{
        loader::{
            sfbinpack::{MoveType, PieceType, TrainingDataEntry},
            DirectSequentialDataLoader, SfBinpackLoader,
        },
        ValueTrainerBuilder,
    },
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use sha2::{Digest, Sha256};

fn tensor_rng(seed: u64, id: &str) -> ChaCha8Rng {
    let mut hash = Sha256::new();
    hash.update(seed.to_le_bytes());
    hash.update(id.as_bytes());
    let digest: [u8; 32] = hash.finalize().into();
    ChaCha8Rng::from_seed(digest)
}

fn write_tensor<W: Write>(
    output: &mut W,
    seed: u64,
    id: &str,
    len: usize,
    stdev: Option<f32>,
) -> io::Result<()> {
    output.write_all(id.as_bytes())?;
    output.write_all(b"\n")?;
    output.write_all(&len.to_le_bytes())?;
    if let Some(stdev) = stdev {
        let normal = Normal::new(0.0_f32, stdev).expect("positive init stdev");
        let mut rng = tensor_rng(seed, id);
        for _ in 0..len {
            output.write_all(&normal.sample(&mut rng).to_le_bytes())?;
        }
    } else {
        let zero = 0.0_f32.to_le_bytes();
        for _ in 0..len {
            output.write_all(&zero)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_seeded_enyo_weights(
    path: &Path,
    seed: u64,
    input_size: usize,
    feature_channels: usize,
    hidden: usize,
    l2_size: usize,
    output_buckets: usize,
    input_factoriser: bool,
    full_heads: bool,
    mixed_activation: bool,
    psqt_residual: bool,
    l0_stdev: f32,
    l1_stdev: f32,
) -> io::Result<()> {
    let mut output = File::create(path)?;
    let head_count = if full_heads { output_buckets } else { 1 };
    write_tensor(&mut output, seed, "l0w", hidden * input_size, Some(l0_stdev))?;
    write_tensor(&mut output, seed, "l0b", hidden, None)?;
    if input_factoriser {
        write_tensor(
            &mut output,
            seed,
            "l0f",
            hidden * feature_channels * 64,
            None,
        )?;
    }
    write_tensor(
        &mut output,
        seed,
        "l1w",
        head_count * l2_size * 2 * hidden,
        Some(l1_stdev),
    )?;
    write_tensor(&mut output, seed, "l1b", head_count * l2_size, None)?;
    write_tensor(
        &mut output,
        seed,
        "l2w",
        head_count * 32 * l2_size,
        Some((2.0 / l2_size as f32).sqrt()),
    )?;
    write_tensor(&mut output, seed, "l2b", head_count * 32, None)?;
    if mixed_activation {
        write_tensor(&mut output, seed, "l2sw", 32 * l2_size, None)?;
        write_tensor(&mut output, seed, "l2sb", 32, None)?;
    }
    write_tensor(
        &mut output,
        seed,
        "l3w",
        output_buckets * 32,
        Some((2.0 / 32.0_f32).sqrt()),
    )?;
    write_tensor(&mut output, seed, "l3b", output_buckets, None)?;
    if psqt_residual {
        write_tensor(
            &mut output,
            seed,
            "psqtw",
            output_buckets * input_size,
            None,
        )?;
        write_tensor(&mut output, seed, "psqtb", output_buckets, None)?;
    }
    Ok(())
}

fn write_seeded_reckless_weights(
    path: &Path,
    seed: u64,
    hidden: usize,
    l2_size: usize,
    output_buckets: usize,
) -> io::Result<()> {
    let mut output = File::create(path)?;
    let input_size = 10 * 12 * 64 + enyo_threats::RECKLESS_DIMENSIONS + 768;
    output.write_all(b"l0w\n")?;
    output.write_all(&(hidden * input_size).to_le_bytes())?;
    let normal = Normal::new(0.0_f32, (2.0 / 32.0_f32).sqrt()).expect("positive stdev");
    let mut rng = tensor_rng(seed, "l0w");
    for _ in 0..hidden * (10 * 12 * 64 + enyo_threats::RECKLESS_DIMENSIONS) {
        output.write_all(&normal.sample(&mut rng).to_le_bytes())?;
    }
    for _ in 0..hidden * 768 {
        output.write_all(&0.0_f32.to_le_bytes())?;
    }
    write_tensor(&mut output, seed, "l0b", hidden, None)?;
    write_tensor(
        &mut output,
        seed,
        "l1w",
        output_buckets * l2_size * hidden,
        Some((2.0 / hidden as f32).sqrt()),
    )?;
    write_tensor(&mut output, seed, "l1b", output_buckets * l2_size, None)?;
    write_tensor(
        &mut output,
        seed,
        "l2w",
        output_buckets * 32 * l2_size,
        Some((2.0 / l2_size as f32).sqrt()),
    )?;
    write_tensor(&mut output, seed, "l2b", output_buckets * 32, None)?;
    write_tensor(
        &mut output,
        seed,
        "l3w",
        output_buckets * 32,
        Some((2.0 / 32.0_f32).sqrt()),
    )?;
    write_tensor(&mut output, seed, "l3b", output_buckets, None)
}

fn env_string(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_owned())
}

fn env_parse<T: std::str::FromStr>(name: &str, default: T) -> T {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn dataset_paths(dataset: &str) -> Vec<String> {
    dataset
        .split(';')
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .map(str::to_owned)
        .collect()
}

fn make_sfbinpack_filter() -> impl Fn(&TrainingDataEntry) -> bool + Clone {
    let min_ply = env_parse("ENYO_BULLET_SFBINPACK_MIN_PLY", 16u16);
    let max_abs_cp = env_parse("ENYO_BULLET_SFBINPACK_MAX_ABS_CP", 10000u32);
    let quiet_only = env_parse("ENYO_BULLET_SFBINPACK_QUIET_ONLY", 1usize) != 0;
    move |entry: &TrainingDataEntry| {
        if entry.ply < min_ply {
            return false;
        }
        if i32::from(entry.score).unsigned_abs() > max_abs_cp {
            return false;
        }
        // Check move quietness before is_checked: the quiet test is a cheap
        // bitboard index while is_checked runs full attack generation.  Non-quiet
        // positions are rejected regardless of check status, so we skip the
        // expensive call for them.
        if quiet_only
            && (entry.mv.mtype() != MoveType::Normal
                || entry.pos.piece_at(entry.mv.to()).piece_type() != PieceType::None)
        {
            return false;
        }
        if entry.pos.is_checked(entry.pos.side_to_move()) {
            return false;
        }
        true
    }
}

fn enyo_affine<'a>(
    builder: &'a ModelBuilder,
    id: &str,
    input_size: usize,
    output_size: usize,
    stdev: f32,
) -> Affine<'a> {
    let weights = builder.new_weights(
        &format!("{id}w"),
        Shape::new(output_size, input_size),
        InitSettings::Normal { mean: 0.0, stdev },
    );
    let bias = builder.new_weights(
        &format!("{id}b"),
        Shape::new(output_size, 1),
        InitSettings::Zeroed,
    );
    Affine { weights, bias }
}

fn zero_affine<'a>(
    builder: &'a ModelBuilder,
    id: &str,
    input_size: usize,
    output_size: usize,
) -> Affine<'a> {
    let weights = builder.new_weights(
        &format!("{id}w"),
        Shape::new(output_size, input_size),
        InitSettings::Zeroed,
    );
    let bias = builder.new_weights(
        &format!("{id}b"),
        Shape::new(output_size, 1),
        InitSettings::Zeroed,
    );
    Affine { weights, bias }
}

fn maybe_frozen<'a, T>(builder: &'a ModelBuilder, frozen: bool, mut f: impl FnMut() -> T) -> T {
    if frozen {
        builder.no_grad(|| f())
    } else {
        f()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct EnyoInputs<
    const INPUT_BUCKETS: usize,
    const FEATURE_CHANNELS: usize,
    const FULL_THREATS: bool,
>;

#[derive(Clone, Copy, Default)]
struct RecklessInputs;

const RECKLESS_BUCKET_LAYOUT: [usize; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 8, 8, 8, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9,
];

impl SparseInputType for RecklessInputs {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        10 * 768 + enyo_threats::RECKLESS_DIMENSIONS + 768
    }

    fn max_active(&self) -> usize {
        64 + enyo_threats::MAX_ACTIVE
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &ChessBoard, mut f: F) {
        let factor_base = 10 * 768 + enyo_threats::RECKLESS_DIMENSIONS;
        ChessBucketsMirrored::new(RECKLESS_BUCKET_LAYOUT).map_features(pos, |stm, ntm| {
            f(stm, ntm);
            f(factor_base + stm % 768, factor_base + ntm % 768);
        });
        let threats = enyo_threats::reckless_active_features(pos);
        assert_eq!(threats[0].len(), threats[1].len());
        for i in 0..threats[0].len() {
            f(10 * 768 + threats[0].get(i), 10 * 768 + threats[1].get(i));
        }
    }

    fn shorthand(&self) -> String {
        "reckless-current".to_string()
    }

    fn description(&self) -> String {
        "Reckless current 10-bucket piece plus 66864 threat inputs".to_string()
    }
}

#[derive(Clone, Copy, Default)]
struct RecklessOutputBuckets;

impl OutputBuckets<ChessBoard> for RecklessOutputBuckets {
    const BUCKETS: usize = 8;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        const LAYOUT: [u8; 33] = [
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4,
            5, 5, 5, 6, 6, 6, 7, 7, 7, 7,
        ];
        LAYOUT[pos.occ().count_ones() as usize]
    }
}

#[rustfmt::skip]
const ENYO_KING_BUCKETS_16: [usize; 64] = [
    15, 15, 14, 14, 14, 14, 15, 15,
    15, 15, 14, 14, 14, 14, 15, 15,
    13, 13, 12, 12, 12, 12, 13, 13,
    13, 13, 12, 12, 12, 12, 13, 13,
    11, 10,  9,  8,  8,  9, 10, 11,
    11, 10,  9,  8,  8,  9, 10, 11,
     7,  6,  5,  4,  4,  5,  6,  7,
     3,  2,  1,  0,  0,  1,  2,  3,
];

#[rustfmt::skip]
const ENYO_KING_BUCKETS_10: [usize; 64] = [
     9,  9,  8,  8,  8,  8,  9,  9,
     9,  9,  8,  8,  8,  8,  9,  9,
     8,  8,  7,  7,  7,  7,  8,  8,
     8,  8,  7,  7,  7,  7,  8,  8,
     6,  6,  5,  5,  5,  5,  6,  6,
     6,  6,  5,  5,  5,  5,  6,  6,
     4,  3,  3,  2,  2,  3,  3,  4,
     1,  1,  0,  0,  0,  0,  1,  1,
];

#[rustfmt::skip]
const ENYO_KING_BUCKETS_32: [usize; 64] = [
    31, 30, 29, 28, 28, 29, 30, 31,
    27, 26, 25, 24, 24, 25, 26, 27,
    23, 22, 21, 20, 20, 21, 22, 23,
    19, 18, 17, 16, 16, 17, 18, 19,
    15, 14, 13, 12, 12, 13, 14, 15,
    11, 10,  9,  8,  8,  9, 10, 11,
     7,  6,  5,  4,  4,  5,  6,  7,
     3,  2,  1,  0,  0,  1,  2,  3,
];

fn enyo_bucket<const INPUT_BUCKETS: usize>(oriented_king_square: usize) -> usize {
    match INPUT_BUCKETS {
        1 | 2 | 4 | 8 | 16 => ENYO_KING_BUCKETS_16[oriented_king_square] * INPUT_BUCKETS / 16,
        10 => ENYO_KING_BUCKETS_10[oriented_king_square],
        32 => ENYO_KING_BUCKETS_32[oriented_king_square],
        _ => panic!("unsupported Enyo input bucket count: {INPUT_BUCKETS}"),
    }
}

fn enyo_feature<const INPUT_BUCKETS: usize, const FEATURE_CHANNELS: usize>(
    piece: u8,
    sq_berserk: u8,
    king_berserk: u8,
    view: usize,
) -> usize {
    let colour = usize::from(piece & 8 != 0);
    let piece_type = usize::from(piece & 7);
    let piece_code = (piece_type << 1) | colour;
    let king = usize::from(king_berserk);
    let sq = usize::from(sq_berserk);
    let flip = 7 * usize::from((king & 4) == 0);
    let orient = flip ^ (56 * view);
    let op = match FEATURE_CHANNELS {
        11 => {
            if piece_type == 5 {
                10
            } else {
                5 * ((piece_code ^ view) & 1) + piece_type
            }
        }
        12 => 6 * ((piece_code ^ view) & 1) + (piece_code >> 1),
        _ => panic!("unsupported Enyo feature channel count: {FEATURE_CHANNELS}"),
    };
    let ok = orient ^ king;
    let osq = orient ^ sq;
    enyo_bucket::<INPUT_BUCKETS>(ok) * FEATURE_CHANNELS * 64 + op * 64 + osq
}

fn bullet_square_to_enyo_net(square: u8) -> u8 {
    square ^ 56
}

impl<const INPUT_BUCKETS: usize, const FEATURE_CHANNELS: usize, const FULL_THREATS: bool>
    SparseInputType for EnyoInputs<INPUT_BUCKETS, FEATURE_CHANNELS, FULL_THREATS>
{
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        INPUT_BUCKETS * FEATURE_CHANNELS * 64
            + if FULL_THREATS {
                enyo_threats::DIMENSIONS
            } else {
                0
            }
    }

    fn max_active(&self) -> usize {
        32 + if FULL_THREATS {
            enyo_threats::MAX_ACTIVE
        } else {
            0
        }
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let stm_king = pos.our_ksq() ^ 56;
        let ntm_king = pos.opp_ksq();
        for (piece, square) in pos.into_iter() {
            let sq = bullet_square_to_enyo_net(square);
            f(
                enyo_feature::<INPUT_BUCKETS, FEATURE_CHANNELS>(piece, sq, stm_king, 0),
                enyo_feature::<INPUT_BUCKETS, FEATURE_CHANNELS>(piece, sq, ntm_king, 1),
            );
        }
        if FULL_THREATS {
            let base = INPUT_BUCKETS * FEATURE_CHANNELS * 64;
            let threats = enyo_threats::active_features(pos);
            assert_eq!(
                threats[0].len(),
                threats[1].len(),
                "FullThreats perspective feature counts differ",
            );
            for i in 0..threats[0].len() {
                f(base + threats[0].get(i), base + threats[1].get(i));
            }
        }
    }

    fn shorthand(&self) -> String {
        format!("enyo-{INPUT_BUCKETS}kb-{FEATURE_CHANNELS}ch")
    }

    fn description(&self) -> String {
        format!("Enyo {INPUT_BUCKETS}-king-bucket {FEATURE_CHANNELS}-channel exported NNUE inputs")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn stm_feature(piece: u8, square: u8, own_king: u8) -> usize {
        enyo_feature::<16, 12>(piece, bullet_square_to_enyo_net(square), own_king ^ 56, 0)
    }

    fn ntm_feature(piece: u8, square: u8, opp_king: u8) -> usize {
        enyo_feature::<16, 12>(piece, bullet_square_to_enyo_net(square), opp_king ^ 56, 1)
    }

    #[test]
    fn enyo_inputs_match_runtime_feature_indices() {
        assert_eq!(bullet_square_to_enyo_net(12), 52); // e2, a1=0 -> a8=0
        assert_eq!(stm_feature(0, 12, 4), 52); // white pawn e2, white view
        assert_eq!(ntm_feature(0, 12, 60), 396); // white pawn e2, black view
        assert_eq!(stm_feature(8, 52, 4), 396); // black pawn e7, white view
        assert_eq!(ntm_feature(8, 52, 60), 52); // black pawn e7, black view
    }

    #[test]
    fn ten_bucket_layout_uses_every_enyo_bucket() {
        let mut used = [false; 10];
        for square in 0..64 {
            let bucket = enyo_bucket::<10>(square);
            assert_eq!(bucket, ENYO_KING_BUCKETS_10[square]);
            used[bucket] = true;
        }
        assert!(used.into_iter().all(|value| value));
    }

    #[test]
    fn seeded_initial_weights_are_reproducible_and_seed_sensitive() {
        let dir = env::temp_dir().join(format!("enyo-seeded-init-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("create seeded-init test directory");
        let first = dir.join("first.bin");
        let same = dir.join("same.bin");
        let different = dir.join("different.bin");
        for (path, seed) in [(&first, 17), (&same, 17), (&different, 18)] {
            write_seeded_enyo_weights(
                path, seed, 24, 12, 8, 4, 2, true, false, false, false, 8.0, 1.0,
            )
            .expect("write seeded weights");
        }
        let first_bytes = fs::read(&first).expect("read first seeded weights");
        assert_eq!(first_bytes, fs::read(&same).expect("read repeated seeded weights"));
        assert_ne!(
            first_bytes,
            fs::read(&different).expect("read different seeded weights")
        );
        fs::remove_dir_all(&dir).expect("remove seeded-init test directory");
    }
}

#[allow(clippy::too_many_arguments)]
fn train_enyo<
    const INPUT_BUCKETS: usize,
    const FEATURE_CHANNELS: usize,
    const OUTPUT_BUCKETS: usize,
    const FULL_THREATS: bool,
    const FULL_HEADS: bool,
    const MIXED_ACTIVATION: bool,
>(
    dataset: String,
    output: String,
    net_id: String,
    hidden: usize,
    l2_size: usize,
    batch_size: usize,
    batches_per_superbatch: usize,
    start_superbatch: usize,
    end_superbatch: usize,
    lr_superbatches: usize,
    threads: usize,
    wdl_proportion: f32,
    initial_lr: f32,
    final_lr: f32,
    l0_stdev: f32,
    l1_stdev: f32,
    l1_export_scale: f32,
    input_factoriser: bool,
    eval_scale: f32,
    save_rate: usize,
    trainable: String,
    weight_decay: f32,
    activation_l1: f32,
    psqt_residual: bool,
) {
    if !matches!(hidden, 512 | 768 | 1024) || l2_size != 16 {
        panic!("Enyo mode supports hidden=512, 768, or 1024 with l2=16");
    }
    if !matches!(INPUT_BUCKETS, 1 | 2 | 4 | 8 | 10 | 16 | 32) {
        panic!("Enyo mode supports only 1, 2, 4, 8, 10, 16, or 32 input king buckets");
    }
    if !matches!(FEATURE_CHANNELS, 11 | 12) {
        panic!("Enyo mode supports only 11 or 12 feature channels");
    }
    if FEATURE_CHANNELS == 11 && !matches!(INPUT_BUCKETS, 10 | 16 | 32) {
        panic!("11-channel Enyo mode requires 10, 16, or 32 input king buckets");
    }
    if !matches!(OUTPUT_BUCKETS, 1 | 2 | 4 | 8) {
        panic!("Enyo mode supports only 1, 2, 4, or 8 output buckets");
    }
    if FULL_HEADS && OUTPUT_BUCKETS == 1 {
        panic!("full-head Enyo mode requires more than one output bucket");
    }
    if FULL_HEADS && FULL_THREATS {
        panic!("full-head and FullThreats changes must be trained separately");
    }
    if MIXED_ACTIVATION && (FULL_HEADS || FULL_THREATS) {
        panic!("mixed activation must be tested independently");
    }
    if psqt_residual && (MIXED_ACTIVATION || FULL_HEADS || FULL_THREATS || OUTPUT_BUCKETS != 8) {
        panic!("PSQT residual requires the shared-head 8-bucket base architecture");
    }
    if start_superbatch == 0 || start_superbatch > end_superbatch {
        panic!("invalid superbatch range {start_superbatch}..={end_superbatch}");
    }
    if lr_superbatches < end_superbatch {
        panic!(
            "LR schedule ends at superbatch {lr_superbatches}, before training ends at {end_superbatch}"
        );
    }
    if !activation_l1.is_finite() || activation_l1 < 0.0 {
        panic!("activation_l1 must be finite and non-negative");
    }
    if FULL_THREATS && input_factoriser {
        panic!("full-threat Enyo mode does not support input_factoriser yet");
    }

    println!("mode=enyo");
    println!("dataset={dataset}");
    println!("output={output}");
    println!("net_id={net_id}");
    println!("hidden={hidden} l2={l2_size}");
    println!("enyo_input_buckets={INPUT_BUCKETS}");
    println!("enyo_feature_channels={FEATURE_CHANNELS}");
    println!("enyo_output_buckets={OUTPUT_BUCKETS}");
    println!("enyo_full_threats={FULL_THREATS}");
    println!("enyo_full_heads={FULL_HEADS}");
    println!("enyo_mixed_activation={MIXED_ACTIVATION}");
    println!("enyo_psqt_residual={psqt_residual}");
    println!("enyo_l0_stdev={l0_stdev} enyo_l1_stdev={l1_stdev}");
    println!("enyo_l1_export_scale={l1_export_scale}");
    println!("enyo_input_factoriser={input_factoriser}");
    println!("enyo_l0_export_scale=1");
    println!("eval_scale={eval_scale}");
    println!("save_rate={save_rate}");
    println!("trainable={trainable}");
    println!("weight_decay={weight_decay}");
    println!("activation_l1={activation_l1}");
    println!(
        "batch_size={batch_size} batches_per_superbatch={batches_per_superbatch} \
         start_superbatch={start_superbatch} end_superbatch={end_superbatch} \
         lr_superbatches={lr_superbatches}"
    );

    if !matches!(
        trainable.as_str(),
        "all" | "input" | "dense-head" | "float-head" | "output" | "squared-branch" | "psqt"
    ) {
        panic!("unsupported trainable mode: {trainable}");
    }
    let train_input = trainable == "all" || trainable == "input";
    let train_l1 = trainable == "all" || trainable == "dense-head";
    let train_l2 = trainable == "all"
        || trainable == "dense-head"
        || trainable == "float-head";
    let train_l3 = trainable == "all"
        || trainable == "dense-head"
        || trainable == "float-head"
        || trainable == "output";
    let train_squared = MIXED_ACTIVATION
        && (trainable == "all" || trainable == "squared-branch");
    if trainable == "squared-branch" && !MIXED_ACTIVATION {
        panic!("squared-branch requires mixed activation");
    }
    let train_psqt = psqt_residual && (trainable == "all" || trainable == "psqt");
    if trainable == "psqt" && !psqt_residual {
        panic!("psqt trainable mode requires PSQT residual architecture");
    }
    if activation_l1 > 0.0 && !train_input {
        panic!("activation_l1 requires a trainable input layer");
    }
    let init_weights = env_string("ENYO_BULLET_INIT_WEIGHTS", "");
    let resume_checkpoint = env_string("ENYO_BULLET_RESUME_CHECKPOINT", "");
    let init_seed = env::var("ENYO_BULLET_INIT_SEED")
        .ok()
        .map(|value| value.parse::<u64>().expect("invalid ENYO_BULLET_INIT_SEED"));
    let export_init_only = env_parse("ENYO_BULLET_EXPORT_INIT_ONLY", 0usize) != 0;

    macro_rules! l0w_format {
        () => {
            if input_factoriser {
                SavedFormat::id("l0w")
                    .transform(|store, weights| {
                        let factoriser = store.get("l0f").values.f32().repeat(INPUT_BUCKETS);
                        weights
                            .into_iter()
                            .zip(factoriser)
                            .map(|(a, b)| a + b)
                            .collect()
                    })
                    .round()
                    .quantise::<i16>(1)
            } else {
                SavedFormat::id("l0w").round().quantise::<i16>(1)
            }
        };
    }

    macro_rules! base_trainer {
        () => {
            if MIXED_ACTIVATION {
                ValueTrainerBuilder::default()
                    .dual_perspective()
                    .optimiser(AdamW)
                    .inputs(EnyoInputs::<INPUT_BUCKETS, FEATURE_CHANNELS, FULL_THREATS>)
                    .save_format(&[
                        l0w_format!(),
                        SavedFormat::id("l0b").round().quantise::<i16>(1),
                        SavedFormat::id("l1w").transpose().round().quantise::<i8>(l1_export_scale as i16),
                        SavedFormat::id("l1b").round().quantise::<i32>(l1_export_scale as i32),
                        SavedFormat::id("l2w").transpose().transform(move |_store, weights| {
                            weights.into_iter().map(|w| w / l1_export_scale).collect()
                        }),
                        SavedFormat::id("l2b"),
                        SavedFormat::id("l2sw").transpose().transform(move |_store, weights| {
                            weights.into_iter().map(|w| w / l1_export_scale).collect()
                        }),
                        SavedFormat::id("l2sb"),
                        SavedFormat::id("l3w").transpose().transform(move |_store, weights| {
                            weights.into_iter().map(|w| w * eval_scale * 32.0).collect()
                        }),
                        SavedFormat::id("l3b").transform(move |_store, weights| {
                            weights.into_iter().map(|w| w * eval_scale * 32.0).collect()
                        }),
                    ])
            } else if psqt_residual {
                ValueTrainerBuilder::default()
                .dual_perspective()
                .optimiser(AdamW)
                .inputs(EnyoInputs::<INPUT_BUCKETS, FEATURE_CHANNELS, FULL_THREATS>)
                .save_format(&[
                    l0w_format!(),
                    SavedFormat::id("l0b").round().quantise::<i16>(1),
                    SavedFormat::id("l1w").transpose().round().quantise::<i8>(l1_export_scale as i16),
                    SavedFormat::id("l1b").round().quantise::<i32>(l1_export_scale as i32),
                    SavedFormat::id("l2w").transpose().transform(move |_store, weights| {
                        weights.into_iter().map(|w| w / l1_export_scale).collect()
                    }),
                    SavedFormat::id("l2b"),
                    SavedFormat::id("l3w").transpose().transform(move |_store, weights| {
                        weights.into_iter().map(|w| w * eval_scale * 32.0).collect()
                    }),
                    SavedFormat::id("l3b").transform(move |_store, weights| {
                        weights.into_iter().map(|w| w * eval_scale * 32.0).collect()
                    }),
                    SavedFormat::id("psqtw").transpose().transform(move |_store, weights| {
                        weights.into_iter().map(|w| w * eval_scale * 32.0).collect()
                    }),
                    SavedFormat::id("psqtb").transform(move |_store, weights| {
                        weights.into_iter().map(|w| w * eval_scale * 32.0).collect()
                    }),
                ])
            } else {
                ValueTrainerBuilder::default()
                .dual_perspective()
                .optimiser(AdamW)
                .inputs(EnyoInputs::<INPUT_BUCKETS, FEATURE_CHANNELS, FULL_THREATS>)
                .save_format(&[
                    l0w_format!(),
                    SavedFormat::id("l0b").round().quantise::<i16>(1),
                    SavedFormat::id("l1w")
                        .transpose()
                        .round()
                        .quantise::<i8>(l1_export_scale as i16),
                    SavedFormat::id("l1b")
                        .round()
                        .quantise::<i32>(l1_export_scale as i32),
                    SavedFormat::id("l2w")
                        .transpose()
                        .transform(move |_store, weights| {
                            weights.into_iter().map(|w| w / l1_export_scale).collect()
                        }),
                    SavedFormat::id("l2b"),
                    SavedFormat::id("l3w")
                        .transpose()
                        .transform(move |_store, weights| {
                            weights.into_iter().map(|w| w * eval_scale * 32.0).collect()
                        }),
                    SavedFormat::id("l3b").transform(move |_store, weights| {
                        weights.into_iter().map(|w| w * eval_scale * 32.0).collect()
                    }),
                ])
            }
        };
    }

    macro_rules! enyo_forward {
        ($builder:expr, $stm_inputs:expr, $ntm_inputs:expr) => {{
            let mut l0 = maybe_frozen($builder, !train_input, || {
                enyo_affine(
                    $builder,
                    "l0",
                    INPUT_BUCKETS * FEATURE_CHANNELS * 64
                        + if FULL_THREATS {
                            enyo_threats::DIMENSIONS
                        } else {
                            0
                        },
                    hidden,
                    l0_stdev,
                )
            });
            if input_factoriser {
                let l0f = maybe_frozen($builder, !train_input, || {
                    $builder.new_weights(
                        "l0f",
                        Shape::new(hidden, FEATURE_CHANNELS * 64),
                        InitSettings::Zeroed,
                    )
                });
                l0.weights = l0.weights + l0f.repeat(INPUT_BUCKETS);
            }
            l0.weights = l0.weights.faux_quantise(1.0, true);
            l0.bias = l0.bias.faux_quantise(1.0, true);

            let mut l1 = maybe_frozen($builder, !train_l1, || {
                enyo_affine($builder, "l1", 2 * hidden, l2_size, l1_stdev)
            });
            l1.weights = l1.weights.faux_quantise(l1_export_scale, true);
            l1.bias = l1.bias.faux_quantise(l1_export_scale, true);
            let l2 = maybe_frozen($builder, !train_l2, || {
                $builder.new_affine("l2", l2_size, 32)
            });
            let l2s = if MIXED_ACTIVATION {
                Some(maybe_frozen($builder, !train_squared, || {
                    zero_affine($builder, "l2s", l2_size, 32)
                }))
            } else {
                None
            };
            let l3 = maybe_frozen($builder, !train_l3, || {
                $builder.new_affine("l3", 32, OUTPUT_BUCKETS)
            });
            let psqt = if psqt_residual {
                Some(maybe_frozen($builder, !train_psqt, || {
                    zero_affine(
                        $builder,
                        "psqt",
                        INPUT_BUCKETS * FEATURE_CHANNELS * 64,
                        OUTPUT_BUCKETS,
                    )
                }))
            } else {
                None
            };

            let stm_hidden = (l0.forward($stm_inputs).max(0.0).min(127.0 * 32.0) / 32.0)
                .faux_quantise(1.0, false);
            let ntm_hidden = (l0.forward($ntm_inputs).max(0.0).min(127.0 * 32.0) / 32.0)
                .faux_quantise(1.0, false);
            let activation_penalty = (stm_hidden.reduce_sum_rows() + ntm_hidden.reduce_sum_rows())
                * (activation_l1 / (2.0 * hidden as f32));
            let x0 = stm_hidden.concat(ntm_hidden);
            let x1_pre = l1.forward(x0);
            let x1 = x1_pre.relu();
            let x2_pre = if let Some(l2s) = l2s {
                let clipped = x1_pre.max(0.0).min(127.0);
                let squared = clipped * clipped / 127.0;
                l2.forward(x1) + l2s.forward(squared)
            } else {
                l2.forward(x1)
            };
            let x2 = x2_pre.relu();
            let output = l3.forward(x2);
            let output = if let Some(psqt) = psqt {
                output + psqt.forward($stm_inputs) - psqt.forward($ntm_inputs)
            } else {
                output
            };
            (output, activation_penalty)
        }};
        ($builder:expr, $stm_inputs:expr, $ntm_inputs:expr, $output_buckets:expr) => {{
            let mut l0 = maybe_frozen($builder, !train_input, || {
                enyo_affine(
                    $builder,
                    "l0",
                    INPUT_BUCKETS * FEATURE_CHANNELS * 64
                        + if FULL_THREATS {
                            enyo_threats::DIMENSIONS
                        } else {
                            0
                        },
                    hidden,
                    l0_stdev,
                )
            });
            if input_factoriser {
                let l0f = maybe_frozen($builder, !train_input, || {
                    $builder.new_weights(
                        "l0f",
                        Shape::new(hidden, FEATURE_CHANNELS * 64),
                        InitSettings::Zeroed,
                    )
                });
                l0.weights = l0.weights + l0f.repeat(INPUT_BUCKETS);
            }
            l0.weights = l0.weights.faux_quantise(1.0, true);
            l0.bias = l0.bias.faux_quantise(1.0, true);

            let head_count = if FULL_HEADS { OUTPUT_BUCKETS } else { 1 };
            let mut l1 = maybe_frozen($builder, !train_l1, || {
                enyo_affine($builder, "l1", 2 * hidden, head_count * l2_size, l1_stdev)
            });
            l1.weights = l1.weights.faux_quantise(l1_export_scale, true);
            l1.bias = l1.bias.faux_quantise(l1_export_scale, true);
            let l2 = maybe_frozen($builder, !train_l2, || {
                $builder.new_affine("l2", l2_size, head_count * 32)
            });
            let l2s = if MIXED_ACTIVATION {
                Some(maybe_frozen($builder, !train_squared, || {
                    zero_affine($builder, "l2s", l2_size, 32)
                }))
            } else {
                None
            };
            let l3 = maybe_frozen($builder, !train_l3, || {
                $builder.new_affine("l3", 32, OUTPUT_BUCKETS)
            });
            let psqt = if psqt_residual {
                Some(maybe_frozen($builder, !train_psqt, || {
                    zero_affine(
                        $builder,
                        "psqt",
                        INPUT_BUCKETS * FEATURE_CHANNELS * 64,
                        OUTPUT_BUCKETS,
                    )
                }))
            } else {
                None
            };

            let stm_hidden = (l0.forward($stm_inputs).max(0.0).min(127.0 * 32.0) / 32.0)
                .faux_quantise(1.0, false);
            let ntm_hidden = (l0.forward($ntm_inputs).max(0.0).min(127.0 * 32.0) / 32.0)
                .faux_quantise(1.0, false);
            let activation_penalty = (stm_hidden.reduce_sum_rows() + ntm_hidden.reduce_sum_rows())
                * (activation_l1 / (2.0 * hidden as f32));
            let x0 = stm_hidden.concat(ntm_hidden);
            let x1_pre = if FULL_HEADS {
                l1.forward(x0).select($output_buckets)
            } else {
                l1.forward(x0)
            };
            let x1 = x1_pre.relu();
            let x2_pre = if FULL_HEADS {
                l2.forward(x1).select($output_buckets)
            } else if let Some(l2s) = l2s {
                let clipped = x1_pre.max(0.0).min(127.0);
                let squared = clipped * clipped / 127.0;
                l2.forward(x1) + l2s.forward(squared)
            } else {
                l2.forward(x1)
            };
            let x2 = x2_pre.relu();
            let output = l3.forward(x2).select($output_buckets);
            let output = if let Some(psqt) = psqt {
                output + (psqt.forward($stm_inputs) - psqt.forward($ntm_inputs))
                    .select($output_buckets)
            } else {
                output
            };
            (output, activation_penalty)
        }};
    }

    macro_rules! run_trainer {
        ($trainer:expr) => {{
            let mut trainer = $trainer;

            let open_params = AdamWParams {
                decay: weight_decay,
                max_weight: 1.0e9,
                min_weight: -1.0e9,
                ..Default::default()
            };
            trainer.optimiser.set_params(open_params);
            if train_input {
                trainer.optimiser.set_params_for_weight(
                    "l0w",
                    AdamWParams {
                        decay: weight_decay,
                        // Enyo stores accumulator weights as int16. Existing Enyo nets
                        // legitimately contain values outside the older +/-4095 guard;
                        // clamping there corrupts the first exported Bullet checkpoint.
                        max_weight: f32::from(i16::MAX),
                        min_weight: f32::from(i16::MIN),
                        ..Default::default()
                    },
                );
                trainer.optimiser.set_params_for_weight(
                    "l0b",
                    AdamWParams {
                        decay: weight_decay,
                        max_weight: f32::from(i16::MAX),
                        min_weight: f32::from(i16::MIN),
                        ..Default::default()
                    },
                );
                if input_factoriser {
                    trainer.optimiser.set_params_for_weight(
                        "l0f",
                        AdamWParams {
                            decay: weight_decay,
                            max_weight: f32::from(i16::MAX),
                            min_weight: f32::from(i16::MIN),
                            ..Default::default()
                        },
                    );
                }
            }
            if train_l1 {
                trainer.optimiser.set_params_for_weight(
                    "l1w",
                    AdamWParams {
                        decay: weight_decay,
                        max_weight: 127.0 / l1_export_scale,
                        min_weight: -128.0 / l1_export_scale,
                        ..Default::default()
                    },
                );
            }

            if !resume_checkpoint.is_empty() {
                if !init_weights.is_empty() {
                    panic!("checkpoint resume and initial weights are mutually exclusive");
                }
                trainer.load_from_checkpoint(&resume_checkpoint);
                println!("loaded_checkpoint={resume_checkpoint}");
            } else if !init_weights.is_empty() {
                trainer
                    .optimiser
                    .load_weights_from_file(&init_weights)
                    .expect("failed to load initial Bullet weights");
                println!("loaded_init_weights={init_weights}");
                trainer.save_to_checkpoint(&format!("{output}/{net_id}-0"));
                if export_init_only {
                    println!("export_init_only=1");
                    return;
                }
            } else if let Some(seed) = init_seed {
                let seeded_path = Path::new(&output).join(format!("{net_id}-seeded-init.bin"));
                write_seeded_enyo_weights(
                    &seeded_path,
                    seed,
                    INPUT_BUCKETS * FEATURE_CHANNELS * 64
                        + if FULL_THREATS {
                            enyo_threats::DIMENSIONS
                        } else {
                            0
                        },
                    FEATURE_CHANNELS,
                    hidden,
                    l2_size,
                    OUTPUT_BUCKETS,
                    input_factoriser,
                    FULL_HEADS,
                    MIXED_ACTIVATION,
                    psqt_residual,
                    l0_stdev,
                    l1_stdev,
                )
                .expect("failed to write deterministic initial weights");
                trainer
                    .optimiser
                    .load_weights_from_file(seeded_path.to_str().expect("UTF-8 init path"))
                    .expect("failed to load deterministic initial weights");
                println!("loaded_init_seed={seed}");
                trainer.save_to_checkpoint(&format!("{output}/{net_id}-0"));
                if export_init_only {
                    println!("export_init_only=1");
                    return;
                }
            } else if export_init_only {
                panic!(
                    "ENYO_BULLET_EXPORT_INIT_ONLY requires ENYO_BULLET_INIT_WEIGHTS or ENYO_BULLET_INIT_SEED"
                );
            }

            let schedule = TrainingSchedule {
                net_id,
                eval_scale,
                steps: TrainingSteps {
                    batch_size,
                    batches_per_superbatch,
                    start_superbatch,
                    end_superbatch,
                },
                wdl_scheduler: wdl::ConstantWDL {
                    value: wdl_proportion,
                },
                lr_scheduler: lr::CosineDecayLR {
                    initial_lr,
                    final_lr,
                    final_superbatch: lr_superbatches,
                },
                save_rate,
            };

            let settings = LocalSettings {
                threads,
                test_set: None,
                output_directory: &output,
                batch_queue_size: 16,
            };

            let loader = env_string("ENYO_BULLET_LOADER", "direct");
            let paths = dataset_paths(&dataset);
            let path_refs = paths.iter().map(String::as_str).collect::<Vec<_>>();
            match loader.as_str() {
                "direct" => {
                    let dataloader = DirectSequentialDataLoader::new(&path_refs);
                    trainer.run(&schedule, &settings, &dataloader);
                }
                "sfbinpack" => {
                    let buffer_mb = env_parse("ENYO_BULLET_SFBINPACK_BUFFER_MB", 1024usize);
                    let dataloader = SfBinpackLoader::new_concat_multiple(
                        &path_refs,
                        buffer_mb,
                        threads,
                        make_sfbinpack_filter(),
                    );
                    trainer.run(&schedule, &settings, &dataloader);
                }
                _ => panic!("unsupported ENYO_BULLET_LOADER={loader}"),
            }
        }};
    }

    if OUTPUT_BUCKETS == 1 {
        run_trainer!(
            base_trainer!().build_custom(|builder, (stm_inputs, ntm_inputs), target| {
                let (output, activation_penalty) = enyo_forward!(builder, stm_inputs, ntm_inputs);
                let loss = output.sigmoid().squared_error(target) + activation_penalty;
                (output, loss)
            })
        );
    } else {
        run_trainer!(base_trainer!()
            .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
            .build_custom(
                |builder, (stm_inputs, ntm_inputs, output_buckets), target| {
                    let (output, activation_penalty) =
                        enyo_forward!(builder, stm_inputs, ntm_inputs, output_buckets);
                    let loss = output.sigmoid().squared_error(target) + activation_penalty;
                    (output, loss)
                }
            ));
    }
}

fn main() {
    let mode = env_string("ENYO_BULLET_MODE", "reckless");
    let dataset = env_string("ENYO_BULLET_DATA", "data/baseline.bullet");
    let output = env_string("ENYO_BULLET_OUT", "checkpoints");
    let net_id = env_string("ENYO_BULLET_NET_ID", "enyo_bullet_spike");
    let hidden = env_parse("ENYO_BULLET_HIDDEN", 1024usize);
    let l2_size = env_parse("ENYO_BULLET_L2", 16usize);
    let batch_size = env_parse("ENYO_BULLET_BATCH_SIZE", 2048usize);
    let batches_per_superbatch = env_parse("ENYO_BULLET_BATCHES", 64usize);
    let start_superbatch = env_parse("ENYO_BULLET_START_SUPERBATCH", 1usize);
    let end_superbatch = env_parse("ENYO_BULLET_SUPERBATCHES", 2048usize);
    let lr_superbatches = env_parse("ENYO_BULLET_LR_SUPERBATCHES", end_superbatch);
    let threads = env_parse("ENYO_BULLET_THREADS", 4usize);
    let wdl_proportion = env_parse("ENYO_BULLET_WDL", 0.75f32);
    let initial_lr = env_parse("ENYO_BULLET_LR", 0.001f32);
    let final_lr = env_parse("ENYO_BULLET_FINAL_LR", initial_lr * 0.3f32);
    let enyo_l0_std = env_parse("ENYO_BULLET_ENYO_L0_STD", 8.0f32);
    let enyo_l1_std = env_parse("ENYO_BULLET_ENYO_L1_STD", 1.0f32);
    let enyo_l1_export_scale = env_parse("ENYO_BULLET_ENYO_L1_EXPORT_SCALE", 1.0f32);
    let enyo_input_factoriser = env_parse("ENYO_BULLET_ENYO_INPUT_FACTORISER", 0usize) != 0;
    let enyo_input_buckets = env_parse("ENYO_BULLET_ENYO_INPUT_BUCKETS", 32usize);
    let enyo_feature_channels = env_parse("ENYO_BULLET_ENYO_FEATURE_CHANNELS", 12usize);
    let enyo_output_buckets = env_parse("ENYO_BULLET_ENYO_OUTPUT_BUCKETS", 1usize);
    let enyo_full_threats = env_parse("ENYO_BULLET_ENYO_FULL_THREATS", 0usize) != 0;
    let enyo_full_heads = env_parse("ENYO_BULLET_ENYO_FULL_HEADS", 0usize) != 0;
    let enyo_mixed_activation =
        env_parse("ENYO_BULLET_ENYO_MIXED_ACTIVATION", 0usize) != 0;
    let enyo_psqt_residual =
        env_parse("ENYO_BULLET_ENYO_PSQT_RESIDUAL", 0usize) != 0;
    let eval_scale = env_parse("ENYO_BULLET_EVAL_SCALE", 400.0f32);
    let save_rate = env_parse("ENYO_BULLET_SAVE_RATE", 64usize);
    let trainable = env_string("ENYO_BULLET_TRAINABLE", "all");
    let weight_decay = env_parse("ENYO_BULLET_WEIGHT_DECAY", 0.0f32);
    let activation_l1 = env_parse("ENYO_BULLET_ACTIVATION_L1", 0.0f32);

    if mode == "enyo" {
        macro_rules! run_enyo {
            ($input_buckets:literal, $feature_channels:literal, $output_buckets:literal, $full_threats:literal, $full_heads:literal, $mixed_activation:literal) => {
                train_enyo::<
                    $input_buckets,
                    $feature_channels,
                    $output_buckets,
                    $full_threats,
                    $full_heads,
                    $mixed_activation,
                >(
                    dataset,
                    output,
                    net_id,
                    hidden,
                    l2_size,
                    batch_size,
                    batches_per_superbatch,
                    start_superbatch,
                    end_superbatch,
                    lr_superbatches,
                    threads,
                    wdl_proportion,
                    initial_lr,
                    final_lr,
                    enyo_l0_std,
                    enyo_l1_std,
                    enyo_l1_export_scale,
                    enyo_input_factoriser,
                    eval_scale,
                    save_rate,
                    trainable,
                    weight_decay,
                    activation_l1,
                    enyo_psqt_residual,
                )
            };
        }

        macro_rules! run_enyo_layout {
            ($input_buckets:literal, $feature_channels:literal) => {
                match (enyo_output_buckets, enyo_full_threats, enyo_full_heads, enyo_mixed_activation, enyo_psqt_residual) {
                    (1, false, false, false, false) => run_enyo!($input_buckets, $feature_channels, 1, false, false, false),
                    (2, false, false, false, false) => run_enyo!($input_buckets, $feature_channels, 2, false, false, false),
                    (4, false, false, false, false) => run_enyo!($input_buckets, $feature_channels, 4, false, false, false),
                    (8, false, false, false, false) => run_enyo!($input_buckets, $feature_channels, 8, false, false, false),
                    (8, false, false, false, true) => run_enyo!($input_buckets, $feature_channels, 8, false, false, false),
                    (8, false, false, true, false) => run_enyo!($input_buckets, $feature_channels, 8, false, false, true),
                    (1, true, false, false, false) => run_enyo!($input_buckets, $feature_channels, 1, true, false, false),
                    (2, true, false, false, false) => run_enyo!($input_buckets, $feature_channels, 2, true, false, false),
                    (4, true, false, false, false) => run_enyo!($input_buckets, $feature_channels, 4, true, false, false),
                    (8, true, false, false, false) => run_enyo!($input_buckets, $feature_channels, 8, true, false, false),
                    (2, false, true, false, false) => run_enyo!($input_buckets, $feature_channels, 2, false, true, false),
                    (4, false, true, false, false) => run_enyo!($input_buckets, $feature_channels, 4, false, true, false),
                    (8, false, true, false, false) => run_enyo!($input_buckets, $feature_channels, 8, false, true, false),
                    _ => {
                        panic!(
                            "unsupported Enyo output/full-threat/full-head combination: \
                             {enyo_output_buckets}/{enyo_full_threats}/{enyo_full_heads}/{enyo_mixed_activation}"
                        )
                    }
                }
            };
        }

        match (enyo_input_buckets, enyo_feature_channels) {
            (1, 12) => run_enyo_layout!(1, 12),
            (2, 12) => run_enyo_layout!(2, 12),
            (4, 12) => run_enyo_layout!(4, 12),
            (8, 12) => run_enyo_layout!(8, 12),
            (10, 12) => run_enyo_layout!(10, 12),
            (16, 12) => run_enyo_layout!(16, 12),
            (32, 12) => run_enyo_layout!(32, 12),
            (10, 11) => run_enyo_layout!(10, 11),
            (16, 11) => run_enyo_layout!(16, 11),
            (32, 11) => run_enyo_layout!(32, 11),
            _ => panic!(
                "unsupported Enyo input layout: buckets={enyo_input_buckets} \
                 channels={enyo_feature_channels}"
            ),
        }
        return;
    }

    const NUM_OUTPUT_BUCKETS: usize = 8;
    const NUM_INPUT_BUCKETS: usize = 10;
    const PIECE_INPUTS: usize = NUM_INPUT_BUCKETS * 768;
    const THREAT_INPUTS: usize = enyo_threats::RECKLESS_DIMENSIONS;
    const FACTOR_INPUTS: usize = 768;
    const TOTAL_INPUTS: usize = PIECE_INPUTS + THREAT_INPUTS + FACTOR_INPUTS;

    println!("mode=reckless");
    println!("dataset={dataset}");
    println!("output={output}");
    println!("net_id={net_id}");
    println!("hidden={hidden} l2={l2_size}");
    println!(
        "batch_size={batch_size} batches_per_superbatch={batches_per_superbatch} superbatches={end_superbatch}"
    );

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(RecklessInputs)
        .output_buckets(RecklessOutputBuckets)
        .save_format(&[
            SavedFormat::id("l0w")
                .transform(move |_store, weights| {
                    let factoriser = weights[(PIECE_INPUTS + THREAT_INPUTS) * hidden..].to_vec();
                    weights[..PIECE_INPUTS * hidden]
                        .iter()
                        .copied()
                        .zip(factoriser.repeat(NUM_INPUT_BUCKETS))
                        .map(|(a, b)| a + b)
                        .collect()
                })
                .round()
                .quantise::<i16>(255),
            SavedFormat::id("l0w")
                .transform(move |_store, weights| {
                    weights[PIECE_INPUTS * hidden..(PIECE_INPUTS + THREAT_INPUTS) * hidden].to_vec()
                })
                .round()
                .quantise::<i8>(255),
            SavedFormat::id("l0b").round().quantise::<i16>(255),
            SavedFormat::id("l1w")
                .transpose()
                .round()
                .quantise::<i8>(64),
            SavedFormat::id("l1b"),
            SavedFormat::id("l2w").transpose(),
            SavedFormat::id("l2b"),
            SavedFormat::id("l3w").transpose(),
            SavedFormat::id("l3b"),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0 = builder.new_affine("l0", TOTAL_INPUTS, hidden);
            l0.init_with_effective_input_size(32);

            let l1 = builder.new_affine("l1", hidden, NUM_OUTPUT_BUCKETS * l2_size);
            let l2 = builder.new_affine("l2", l2_size, NUM_OUTPUT_BUCKETS * 32);
            let l3 = builder.new_affine("l3", 32, NUM_OUTPUT_BUCKETS);

            let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let hl1 = stm_hidden.concat(ntm_hidden);
            let hl2 = l1.forward(hl1).select(output_buckets).screlu();
            let hl3 = l2.forward(hl2).select(output_buckets).screlu();
            l3.forward(hl3).select(output_buckets)
        });

    let stricter_clipping = AdamWParams {
        max_weight: 0.99,
        min_weight: -0.99,
        ..Default::default()
    };
    trainer
        .optimiser
        .set_params_for_weight("l0w", stricter_clipping);

    let resume_checkpoint = env_string("ENYO_BULLET_RESUME_CHECKPOINT", "");
    let init_weights = env_string("ENYO_BULLET_INIT_WEIGHTS", "");
    let init_seed = env::var("ENYO_BULLET_INIT_SEED")
        .ok()
        .and_then(|value| value.parse::<u64>().ok());
    if !resume_checkpoint.is_empty() {
        if !init_weights.is_empty() {
            panic!("checkpoint resume and initial weights are mutually exclusive");
        }
        trainer.load_from_checkpoint(&resume_checkpoint);
        println!("loaded_checkpoint={resume_checkpoint}");
    } else if !init_weights.is_empty() {
        trainer
            .optimiser
            .load_weights_from_file(&init_weights)
            .expect("failed to load initial Reckless weights");
        println!("loaded_init_weights={init_weights}");
    } else if let Some(seed) = init_seed {
        let seeded_path = Path::new(&output).join(format!("{net_id}-seeded-init.bin"));
        write_seeded_reckless_weights(
            &seeded_path,
            seed,
            hidden,
            l2_size,
            NUM_OUTPUT_BUCKETS,
        )
        .expect("failed to write deterministic Reckless initial weights");
        trainer
            .optimiser
            .load_weights_from_file(seeded_path.to_str().expect("UTF-8 init path"))
            .expect("failed to load deterministic Reckless initial weights");
        println!("loaded_init_seed={seed}");
        trainer.save_to_checkpoint(&format!("{output}/{net_id}-0"));
    }

    let schedule = TrainingSchedule {
        net_id,
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size,
            batches_per_superbatch,
            start_superbatch: 1,
            end_superbatch,
        },
        wdl_scheduler: wdl::ConstantWDL {
            value: wdl_proportion,
        },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr,
            final_lr,
            final_superbatch: end_superbatch,
        },
        save_rate,
    };

    let settings = LocalSettings {
        threads,
        test_set: None,
        output_directory: &output,
        batch_queue_size: 16,
    };

    let loader = env_string("ENYO_BULLET_LOADER", "direct");
    let paths = dataset_paths(&dataset);
    let path_refs = paths.iter().map(String::as_str).collect::<Vec<_>>();
    match loader.as_str() {
        "direct" => {
            let dataloader = DirectSequentialDataLoader::new(&path_refs);
            trainer.run(&schedule, &settings, &dataloader);
        }
        "sfbinpack" => {
            let buffer_mb = env_parse("ENYO_BULLET_SFBINPACK_BUFFER_MB", 1024usize);
            let dataloader = SfBinpackLoader::new_concat_multiple(
                &path_refs,
                buffer_mb,
                threads,
                make_sfbinpack_filter(),
            );
            trainer.run(&schedule, &settings, &dataloader);
        }
        _ => panic!("unsupported ENYO_BULLET_LOADER={loader}"),
    }
}
