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

fn sfbinpack_filter(entry: &TrainingDataEntry) -> bool {
    let min_ply = env_parse("ENYO_BULLET_SFBINPACK_MIN_PLY", 16u16);
    let max_abs_cp = env_parse("ENYO_BULLET_SFBINPACK_MAX_ABS_CP", 10000u32);
    let quiet_only = env_parse("ENYO_BULLET_SFBINPACK_QUIET_ONLY", 1usize) != 0;

    if entry.ply < min_ply {
        return false;
    }
    if i32::from(entry.score).unsigned_abs() > max_abs_cp {
        return false;
    }
    if entry.pos.is_checked(entry.pos.side_to_move()) {
        return false;
    }
    if quiet_only {
        return entry.mv.mtype() == MoveType::Normal
            && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None;
    }
    true
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
struct EnyoInputs;

#[rustfmt::skip]
const ENYO_KING_BUCKETS: [usize; 64] = [
    15, 15, 14, 14, 14, 14, 15, 15,
    15, 15, 14, 14, 14, 14, 15, 15,
    13, 13, 12, 12, 12, 12, 13, 13,
    13, 13, 12, 12, 12, 12, 13, 13,
    11, 10,  9,  8,  8,  9, 10, 11,
    11, 10,  9,  8,  8,  9, 10, 11,
     7,  6,  5,  4,  4,  5,  6,  7,
     3,  2,  1,  0,  0,  1,  2,  3,
];

fn enyo_feature(piece: u8, sq_berserk: u8, king_berserk: u8, view: usize) -> usize {
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
    ENYO_KING_BUCKETS[ok] * 12 * 64 + op * 64 + osq
}

impl SparseInputType for EnyoInputs {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        16 * 12 * 64
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
                enyo_feature(piece, sq, stm_king, 0),
                enyo_feature(piece, sq, ntm_king, 1),
            );
        }
    }

    fn shorthand(&self) -> String {
        "enyo-16kb".to_string()
    }

    fn description(&self) -> String {
        "Enyo 16-king-bucket exported NNUE inputs".to_string()
    }
}

#[allow(clippy::too_many_arguments)]
fn train_enyo(
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
    eval_scale: f32,
    save_rate: usize,
    trainable: String,
    weight_decay: f32,
) {
    if hidden != 1024 || l2_size != 16 {
        panic!("Enyo mode writes the fixed Enyo .nn layout; hidden=1024 and l2=16 are required");
    }

    println!("mode=enyo");
    println!("dataset={dataset}");
    println!("output={output}");
    println!("net_id={net_id}");
    println!("hidden={hidden} l2={l2_size}");
    println!("enyo_l0_stdev={l0_stdev} enyo_l1_stdev={l1_stdev}");
    println!("enyo_l1_export_scale={l1_export_scale}");
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

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(EnyoInputs)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(1),
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
        .build(|builder, stm_inputs, ntm_inputs| {
            let mut l0 = maybe_frozen(builder, !train_input, || {
                enyo_affine(builder, "l0", 16 * 12 * 64, hidden, l0_stdev)
            });
            l0.weights = l0.weights.faux_quantise(1.0, true);
            l0.bias = l0.bias.faux_quantise(1.0, true);

            let mut l1 = maybe_frozen(builder, !train_l1, || {
                enyo_affine(builder, "l1", 2 * hidden, l2_size, l1_stdev)
            });
            l1.weights = l1.weights.faux_quantise(l1_export_scale, true);
            l1.bias = l1.bias.faux_quantise(l1_export_scale, true);
            let l2 = maybe_frozen(builder, !train_l2, || builder.new_affine("l2", l2_size, 32));
            let l3 = maybe_frozen(builder, !train_l3, || builder.new_affine("l3", 32, 1));

            let stm_hidden = (l0.forward(stm_inputs).max(0.0).min(127.0 * 32.0) / 32.0)
                .faux_quantise(1.0, false);
            let ntm_hidden = (l0.forward(ntm_inputs).max(0.0).min(127.0 * 32.0) / 32.0)
                .faux_quantise(1.0, false);
            let x0 = stm_hidden.concat(ntm_hidden);
            let x1 = l1.forward(x0).relu();
            let x2 = l2.forward(x1).relu();
            l3.forward(x2)
        });

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
                max_weight: 4095.0,
                min_weight: -4095.0,
                ..Default::default()
            },
        );
        trainer.optimiser.set_params_for_weight(
            "l0b",
            AdamWParams {
                decay: weight_decay,
                max_weight: 4095.0,
                min_weight: -4095.0,
                ..Default::default()
            },
        );
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
            let dataloader =
                SfBinpackLoader::new_concat_multiple(&path_refs, buffer_mb, threads, sfbinpack_filter);
            trainer.run(&schedule, &settings, &dataloader);
        }
        _ => panic!("unsupported ENYO_BULLET_LOADER={loader}"),
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
    let end_superbatch = env_parse("ENYO_BULLET_SUPERBATCHES", 2usize);
    let threads = env_parse("ENYO_BULLET_THREADS", 4usize);
    let wdl_proportion = env_parse("ENYO_BULLET_WDL", 0.75f32);
    let initial_lr = env_parse("ENYO_BULLET_LR", 0.001f32);
    let final_lr = env_parse("ENYO_BULLET_FINAL_LR", initial_lr * 0.3f32);
    let enyo_l0_std = env_parse("ENYO_BULLET_ENYO_L0_STD", 8.0f32);
    let enyo_l1_std = env_parse("ENYO_BULLET_ENYO_L1_STD", 1.0f32);
    let enyo_l1_export_scale = env_parse("ENYO_BULLET_ENYO_L1_EXPORT_SCALE", 1.0f32);
    let eval_scale = env_parse("ENYO_BULLET_EVAL_SCALE", 400.0f32);
    let save_rate = env_parse("ENYO_BULLET_SAVE_RATE", 1usize);
    let trainable = env_string("ENYO_BULLET_TRAINABLE", "all");
    let weight_decay = env_parse("ENYO_BULLET_WEIGHT_DECAY", 0.0f32);

    if mode == "enyo" {
        train_enyo(
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
            eval_scale,
            save_rate,
            trainable,
            weight_decay,
        );
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
            let dataloader =
                SfBinpackLoader::new_concat_multiple(&path_refs, buffer_mb, threads, sfbinpack_filter);
            trainer.run(&schedule, &settings, &dataloader);
        }
        _ => panic!("unsupported ENYO_BULLET_LOADER={loader}"),
    }
}
