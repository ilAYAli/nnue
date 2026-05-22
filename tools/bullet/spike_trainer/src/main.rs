use std::env;

use bullet_lib::{
    game::{
        formats::bulletformat::ChessBoard,
        inputs::{ChessBucketsMirrored, SparseInputType, get_num_buckets},
        outputs::MaterialCount,
    },
    nn::{
        InitSettings, Shape,
        optimiser::{AdamW, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::DirectSequentialDataLoader},
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

#[derive(Clone, Copy, Debug, Default)]
struct EnyoInputs;

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
        32 * 12 * 64
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
        "enyo-32kb".to_string()
    }

    fn description(&self) -> String {
        "Enyo 32-king-bucket exported NNUE inputs".to_string()
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
) {
    if hidden != 1024 || l2_size != 16 {
        panic!("Enyo mode writes the fixed Enyo .nn layout; hidden=1024 and l2=16 are required");
    }

    println!("mode=enyo");
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
        .inputs(EnyoInputs)
        .save_format(&[
            SavedFormat::id("l0w").transpose().round().quantise::<i16>(1),
            SavedFormat::id("l0b").round().quantise::<i16>(1),
            SavedFormat::id("l1w").transpose().round().quantise::<i8>(1),
            SavedFormat::id("l1b").round().quantise::<i32>(1),
            SavedFormat::id("l2w").transpose(),
            SavedFormat::id("l2b"),
            SavedFormat::id("l3w").transpose(),
            SavedFormat::id("l3b"),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            let l0 = builder.new_affine("l0", 32 * 12 * 64, hidden);
            l0.init_with_effective_input_size(32);
            let l1 = builder.new_affine("l1", 2 * hidden, l2_size);
            let l2 = builder.new_affine("l2", l2_size, 32);
            let l3 = builder.new_affine("l3", 32, 1);

            let stm_hidden = l0.forward(stm_inputs).max(0.0).min(127.0 * 32.0) / 32.0;
            let ntm_hidden = l0.forward(ntm_inputs).max(0.0).min(127.0 * 32.0) / 32.0;
            let x0 = stm_hidden.concat(ntm_hidden);
            let x1 = l1.forward(x0).relu();
            let x2 = l2.forward(x1).relu();
            l3.forward(x2) / (400.0 * 32.0)
        });

    trainer.optimiser.set_params_for_weight(
        "l0w",
        AdamWParams { max_weight: 4095.0, min_weight: -4095.0, ..Default::default() },
    );
    trainer.optimiser.set_params_for_weight(
        "l0b",
        AdamWParams { max_weight: 4095.0, min_weight: -4095.0, ..Default::default() },
    );
    trainer.optimiser.set_params_for_weight(
        "l1w",
        AdamWParams { max_weight: 127.0, min_weight: -128.0, ..Default::default() },
    );

    let schedule = TrainingSchedule {
        net_id,
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size,
            batches_per_superbatch,
            start_superbatch: 1,
            end_superbatch,
        },
        wdl_scheduler: wdl::ConstantWDL { value: wdl_proportion },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr,
            final_lr,
            final_superbatch: end_superbatch,
        },
        save_rate: 1,
    };

    let settings = LocalSettings {
        threads,
        test_set: None,
        output_directory: &output,
        batch_queue_size: 16,
    };

    let dataloader = DirectSequentialDataLoader::new(&[&dataset]);
    trainer.run(&schedule, &settings, &dataloader);
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
        );
        return;
    }

    const NUM_OUTPUT_BUCKETS: usize = 8;
    #[rustfmt::skip]
    const BUCKET_LAYOUT: [usize; 32] = [
        0, 1, 2, 3,
        4, 4, 5, 5,
        6, 6, 6, 6,
        7, 7, 7, 7,
        8, 8, 8, 8,
        8, 8, 8, 8,
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
        save_rate: 1,
    };

    let settings = LocalSettings {
        threads,
        test_set: None,
        output_directory: &output,
        batch_queue_size: 16,
    };

    let dataloader = DirectSequentialDataLoader::new(&[&dataset]);
    trainer.run(&schedule, &settings, &dataloader);
}
