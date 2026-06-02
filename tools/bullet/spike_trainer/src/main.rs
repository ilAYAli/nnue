use std::env;

use bullet_lib::{
    game::{
        formats::bulletformat::ChessBoard,
        inputs::{ChessBucketsMirrored, SparseInputType, get_num_buckets},
        outputs::MaterialCount,
    },
    nn::{
        Affine, InitSettings, ModelBuilder, Shape,
        optimiser::{AdamW, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{
        ValueTrainerBuilder,
        loader::{
            DirectSequentialDataLoader, SfBinpackLoader,
            sfbinpack::{MoveType, PieceType, TrainingDataEntry},
        },
    },
};

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

fn maybe_frozen<'a, T>(builder: &'a ModelBuilder, frozen: bool, mut f: impl FnMut() -> T) -> T {
    if frozen { builder.no_grad(|| f()) } else { f() }
}

#[derive(Clone, Copy, Debug, Default)]
struct EnyoInputs<const INPUT_BUCKETS: usize>;

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
        32 => ENYO_KING_BUCKETS_32[oriented_king_square],
        _ => panic!("unsupported Enyo input bucket count: {INPUT_BUCKETS}"),
    }
}

fn enyo_feature<const INPUT_BUCKETS: usize>(
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
    let op = 6 * ((piece_code ^ view) & 1) + (piece_code >> 1);
    let ok = orient ^ king;
    let osq = orient ^ sq;
    enyo_bucket::<INPUT_BUCKETS>(ok) * 12 * 64 + op * 64 + osq
}

impl<const INPUT_BUCKETS: usize> SparseInputType for EnyoInputs<INPUT_BUCKETS> {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        INPUT_BUCKETS * 12 * 64
    }

    fn max_active(&self) -> usize {
        32
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let stm_king = pos.our_ksq() ^ 56;
        let ntm_king = pos.opp_ksq();
        for (piece, square) in pos.into_iter() {
            let sq = square ^ 56;
            f(
                enyo_feature::<INPUT_BUCKETS>(piece, sq, stm_king, 0),
                enyo_feature::<INPUT_BUCKETS>(piece, sq, ntm_king, 1),
            );
        }
    }

    fn shorthand(&self) -> String {
        format!("enyo-{INPUT_BUCKETS}kb")
    }

    fn description(&self) -> String {
        format!("Enyo {INPUT_BUCKETS}-king-bucket exported NNUE inputs")
    }
}

#[allow(clippy::too_many_arguments)]
fn train_enyo<const INPUT_BUCKETS: usize, const OUTPUT_BUCKETS: usize>(
    dataset: String,
    output: String,
    net_id: String,
    hidden: usize,
    l2_size: usize,
    batch_size: usize,
    batches_per_superbatch: usize,
    end_superbatch: usize,
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
) {
    if !matches!(hidden, 1024 | 1280) || l2_size != 16 {
        panic!(
            "Enyo mode supports native layout hidden=1024 or 1280 with l2=16; \
             engine parity/NPS support is required before using non-1024 nets"
        );
    }
    if !matches!(INPUT_BUCKETS, 1 | 2 | 4 | 8 | 16 | 32) {
        panic!("Enyo mode supports only 1, 2, 4, 8, 16, or 32 input king buckets");
    }
    if !matches!(OUTPUT_BUCKETS, 1 | 2 | 4 | 8) {
        panic!("Enyo mode supports only 1, 2, 4, or 8 output buckets");
    }

    println!("mode=enyo");
    println!("dataset={dataset}");
    println!("output={output}");
    println!("net_id={net_id}");
    println!("hidden={hidden} l2={l2_size}");
    println!("enyo_input_buckets={INPUT_BUCKETS}");
    println!("enyo_output_buckets={OUTPUT_BUCKETS}");
    println!("enyo_l0_stdev={l0_stdev} enyo_l1_stdev={l1_stdev}");
    println!("enyo_l1_export_scale={l1_export_scale}");
    println!("enyo_input_factoriser={input_factoriser}");
    println!("enyo_l0_export_scale=1");
    println!("eval_scale={eval_scale}");
    println!("save_rate={save_rate}");
    println!("trainable={trainable}");
    println!("weight_decay={weight_decay}");
    println!(
        "batch_size={batch_size} batches_per_superbatch={batches_per_superbatch} superbatches={end_superbatch}"
    );

    let train_input = trainable == "all" || trainable == "input";
    let train_l1 = trainable == "all";
    let train_l2 = trainable == "all" || trainable == "float-head";
    let train_l3 = trainable == "all" || trainable == "float-head" || trainable == "output";
    let init_weights = env_string("ENYO_BULLET_INIT_WEIGHTS", "");
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
            ValueTrainerBuilder::default()
                .dual_perspective()
                .optimiser(AdamW)
                .inputs(EnyoInputs::<INPUT_BUCKETS>)
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
                .loss_fn(|output, target| output.sigmoid().squared_error(target))
        };
    }

    macro_rules! enyo_forward {
        ($builder:expr, $stm_inputs:expr, $ntm_inputs:expr) => {{
            let mut l0 = maybe_frozen($builder, !train_input, || {
                enyo_affine($builder, "l0", INPUT_BUCKETS * 12 * 64, hidden, l0_stdev)
            });
            if input_factoriser {
                let l0f = maybe_frozen($builder, !train_input, || {
                    $builder.new_weights("l0f", Shape::new(hidden, 12 * 64), InitSettings::Zeroed)
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
            let l3 = maybe_frozen($builder, !train_l3, || {
                $builder.new_affine("l3", 32, OUTPUT_BUCKETS)
            });

            let stm_hidden = (l0.forward($stm_inputs).max(0.0).min(127.0 * 32.0) / 32.0)
                .faux_quantise(1.0, false);
            let ntm_hidden = (l0.forward($ntm_inputs).max(0.0).min(127.0 * 32.0) / 32.0)
                .faux_quantise(1.0, false);
            let x0 = stm_hidden.concat(ntm_hidden);
            let x1 = l1.forward(x0).relu();
            let x2 = l2.forward(x1).relu();
            l3.forward(x2)
        }};
        ($builder:expr, $stm_inputs:expr, $ntm_inputs:expr, $output_buckets:expr) => {{ enyo_forward!($builder, $stm_inputs, $ntm_inputs).select($output_buckets) }};
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

            if !init_weights.is_empty() {
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
            } else if export_init_only {
                panic!("ENYO_BULLET_EXPORT_INIT_ONLY requires ENYO_BULLET_INIT_WEIGHTS");
            }

            let schedule = TrainingSchedule {
                net_id,
                eval_scale,
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
        }};
    }

    if OUTPUT_BUCKETS == 1 {
        run_trainer!(base_trainer!().build(|builder, stm_inputs, ntm_inputs| {
            enyo_forward!(builder, stm_inputs, ntm_inputs)
        }));
    } else {
        run_trainer!(
            base_trainer!()
                .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
                .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
                    enyo_forward!(builder, stm_inputs, ntm_inputs, output_buckets)
                })
        );
    }
}

fn main() {
    let mode = env_string("ENYO_BULLET_MODE", "reckless");
    let dataset = env_string("ENYO_BULLET_DATA", "data/baseline.data");
    let output = env_string("ENYO_BULLET_OUT", "checkpoints");
    let net_id = env_string("ENYO_BULLET_NET_ID", "enyo_bullet_spike");
    let hidden = env_parse("ENYO_BULLET_HIDDEN", 1024usize);
    let l2_size = env_parse("ENYO_BULLET_L2", 16usize);
    let batch_size = env_parse("ENYO_BULLET_BATCH_SIZE", 2048usize);
    let batches_per_superbatch = env_parse("ENYO_BULLET_BATCHES", 64usize);
    let end_superbatch = env_parse("ENYO_BULLET_SUPERBATCHES", 2048usize);
    let threads = env_parse("ENYO_BULLET_THREADS", 4usize);
    let wdl_proportion = env_parse("ENYO_BULLET_WDL", 0.75f32);
    let initial_lr = env_parse("ENYO_BULLET_LR", 0.001f32);
    let final_lr = env_parse("ENYO_BULLET_FINAL_LR", initial_lr * 0.3f32);
    let enyo_l0_std = env_parse("ENYO_BULLET_ENYO_L0_STD", 8.0f32);
    let enyo_l1_std = env_parse("ENYO_BULLET_ENYO_L1_STD", 1.0f32);
    let enyo_l1_export_scale = env_parse("ENYO_BULLET_ENYO_L1_EXPORT_SCALE", 1.0f32);
    let enyo_input_factoriser = env_parse("ENYO_BULLET_ENYO_INPUT_FACTORISER", 0usize) != 0;
    let enyo_input_buckets = env_parse("ENYO_BULLET_ENYO_INPUT_BUCKETS", 32usize);
    let enyo_output_buckets = env_parse("ENYO_BULLET_ENYO_OUTPUT_BUCKETS", 1usize);
    let eval_scale = env_parse("ENYO_BULLET_EVAL_SCALE", 400.0f32);
    let save_rate = env_parse("ENYO_BULLET_SAVE_RATE", 64usize);
    let trainable = env_string("ENYO_BULLET_TRAINABLE", "all");
    let weight_decay = env_parse("ENYO_BULLET_WEIGHT_DECAY", 0.0f32);

    if mode == "enyo" {
        macro_rules! run_enyo {
            ($input_buckets:literal, $output_buckets:literal) => {
                train_enyo::<$input_buckets, $output_buckets>(
                    dataset,
                    output,
                    net_id,
                    hidden,
                    l2_size,
                    batch_size,
                    batches_per_superbatch,
                    end_superbatch,
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
                )
            };
        }

        macro_rules! run_enyo_input {
            ($input_buckets:literal) => {
                match enyo_output_buckets {
                    1 => run_enyo!($input_buckets, 1),
                    2 => run_enyo!($input_buckets, 2),
                    4 => run_enyo!($input_buckets, 4),
                    8 => run_enyo!($input_buckets, 8),
                    _ => {
                        panic!("unsupported ENYO_BULLET_ENYO_OUTPUT_BUCKETS={enyo_output_buckets}")
                    }
                }
            };
        }

        match enyo_input_buckets {
            1 => run_enyo_input!(1),
            2 => run_enyo_input!(2),
            4 => run_enyo_input!(4),
            8 => run_enyo_input!(8),
            16 => run_enyo_input!(16),
            32 => run_enyo_input!(32),
            _ => panic!("unsupported ENYO_BULLET_ENYO_INPUT_BUCKETS={enyo_input_buckets}"),
        }
        return;
    }

    const NUM_OUTPUT_BUCKETS: usize = 8;
    #[rustfmt::skip]
    const BUCKET_LAYOUT: [usize; 32] = [
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 8, 8, 8,
        9, 9, 9, 9,
        9, 9, 9, 9,
        9, 9, 9, 9,
        9, 9, 9, 9,
        9, 9, 9, 9,
    ];
    const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);

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
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("l0w")
                .transform(|store, weights| {
                    let factoriser = store.get("l0f").values.f32().repeat(NUM_INPUT_BUCKETS);
                    weights
                        .into_iter()
                        .zip(factoriser)
                        .map(|(a, b)| a + b)
                        .collect()
                })
                .round()
                .quantise::<i16>(255),
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
            let l0f = builder.new_weights("l0f", Shape::new(hidden, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, hidden);
            l0.init_with_effective_input_size(32);
            l0.weights = l0.weights + expanded_factoriser;

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
    trainer
        .optimiser
        .set_params_for_weight("l0f", stricter_clipping);

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
