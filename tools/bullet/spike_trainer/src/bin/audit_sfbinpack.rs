use std::{env, process};

use bullet_lib::{
    game::formats::bulletformat::ChessBoard,
    value::loader::{
        DataLoader, GameResult, LoadableDataType, SfBinpackLoader,
        sfbinpack::{MoveType, PieceType, TrainingDataEntry},
    },
};

#[derive(Clone, Copy)]
struct Filter {
    min_ply: u16,
    max_abs_cp: u32,
    quiet_only: bool,
}

impl Filter {
    fn keep(self, entry: &TrainingDataEntry) -> bool {
        if entry.ply < self.min_ply {
            return false;
        }
        if i32::from(entry.score).unsigned_abs() > self.max_abs_cp {
            return false;
        }
        if entry.pos.is_checked(entry.pos.side_to_move()) {
            return false;
        }
        if self.quiet_only {
            return entry.mv.mtype() == MoveType::Normal
                && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None;
        }
        true
    }
}

#[derive(Default)]
struct Stats {
    rows: u64,
    wins: u64,
    draws: u64,
    losses: u64,
    score_sum: i64,
    abs_score_sum: u64,
    score_result_agree: u64,
    non_draws: u64,
    buckets: [Bucket; 6],
}

#[derive(Clone, Copy, Default)]
struct Bucket {
    rows: u64,
    wins: u64,
    draws: u64,
    losses: u64,
    score_sum: i64,
    abs_score_sum: u64,
    score_result_agree: u64,
    non_draws: u64,
}

fn bucket_index(abs_cp: u32) -> usize {
    match abs_cp {
        0..=50 => 0,
        51..=100 => 1,
        101..=300 => 2,
        301..=800 => 3,
        801..=1600 => 4,
        _ => 5,
    }
}

fn bucket_name(idx: usize) -> &'static str {
    match idx {
        0 => "0-50",
        1 => "51-100",
        2 => "101-300",
        3 => "301-800",
        4 => "801-1600",
        _ => "1601+",
    }
}

fn add_bucket(bucket: &mut Bucket, score: i16, result: GameResult) {
    bucket.rows += 1;
    bucket.score_sum += i64::from(score);
    bucket.abs_score_sum += i32::from(score).unsigned_abs() as u64;
    match result {
        GameResult::Win => bucket.wins += 1,
        GameResult::Draw => bucket.draws += 1,
        GameResult::Loss => bucket.losses += 1,
    }
    if result != GameResult::Draw {
        bucket.non_draws += 1;
        if (score > 0 && result == GameResult::Win) || (score < 0 && result == GameResult::Loss) {
            bucket.score_result_agree += 1;
        }
    }
}

fn add(stats: &mut Stats, board: &ChessBoard) {
    let score = board.score();
    let result = board.result();
    stats.rows += 1;
    stats.score_sum += i64::from(score);
    stats.abs_score_sum += i32::from(score).unsigned_abs() as u64;
    match result {
        GameResult::Win => stats.wins += 1,
        GameResult::Draw => stats.draws += 1,
        GameResult::Loss => stats.losses += 1,
    }
    if result != GameResult::Draw {
        stats.non_draws += 1;
        if (score > 0 && result == GameResult::Win) || (score < 0 && result == GameResult::Loss) {
            stats.score_result_agree += 1;
        }
    }
    add_bucket(
        &mut stats.buckets[bucket_index(i32::from(score).unsigned_abs())],
        score,
        result,
    );
}

fn pct(n: u64, d: u64) -> f64 {
    if d == 0 {
        0.0
    } else {
        100.0 * n as f64 / d as f64
    }
}

fn mean(sum: i64, rows: u64) -> f64 {
    if rows == 0 {
        0.0
    } else {
        sum as f64 / rows as f64
    }
}

fn mean_abs(sum: u64, rows: u64) -> f64 {
    if rows == 0 {
        0.0
    } else {
        sum as f64 / rows as f64
    }
}

fn usage() -> ! {
    eprintln!(
        "usage: audit_sfbinpack --data PATH[;PATH...] [--limit N] [--buffer-mb N] \
         [--threads N] [--min-ply N] [--max-abs-cp N] [--quiet-only 0|1]"
    );
    process::exit(2);
}

fn main() {
    let mut data = String::new();
    let mut limit = 1_000_000_u64;
    let mut buffer_mb = 1024_usize;
    let mut threads = 4_usize;
    let mut filter = Filter {
        min_ply: 16,
        max_abs_cp: 10000,
        quiet_only: true,
    };

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| usage());
        match arg.as_str() {
            "--data" => data = value(),
            "--limit" => limit = value().parse().unwrap_or_else(|_| usage()),
            "--buffer-mb" => buffer_mb = value().parse().unwrap_or_else(|_| usage()),
            "--threads" => threads = value().parse().unwrap_or_else(|_| usage()),
            "--min-ply" => filter.min_ply = value().parse().unwrap_or_else(|_| usage()),
            "--max-abs-cp" => filter.max_abs_cp = value().parse().unwrap_or_else(|_| usage()),
            "--quiet-only" => filter.quiet_only = value() != "0",
            _ => usage(),
        }
    }

    if data.is_empty() {
        usage();
    }

    let paths = data
        .split(';')
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .collect::<Vec<_>>();
    let loader = SfBinpackLoader::new_concat_multiple(&paths, buffer_mb, threads, move |entry| {
        filter.keep(entry)
    });
    let mut stats = Stats::default();
    loader.map_chunks(0, |chunk| {
        for board in chunk {
            add(&mut stats, board);
            if stats.rows >= limit {
                return true;
            }
        }
        false
    });

    println!("rows={}", stats.rows);
    println!(
        "result win={} ({:.2}%) draw={} ({:.2}%) loss={} ({:.2}%)",
        stats.wins,
        pct(stats.wins, stats.rows),
        stats.draws,
        pct(stats.draws, stats.rows),
        stats.losses,
        pct(stats.losses, stats.rows)
    );
    println!(
        "score mean={:.2} mean_abs={:.2} score_result_agree={}/{} ({:.2}%)",
        mean(stats.score_sum, stats.rows),
        mean_abs(stats.abs_score_sum, stats.rows),
        stats.score_result_agree,
        stats.non_draws,
        pct(stats.score_result_agree, stats.non_draws)
    );
    println!("bucket rows win% draw% loss% mean_cp mean_abs_cp agree_non_draw%");
    for (idx, bucket) in stats.buckets.iter().enumerate() {
        println!(
            "{} {} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2}",
            bucket_name(idx),
            bucket.rows,
            pct(bucket.wins, bucket.rows),
            pct(bucket.draws, bucket.rows),
            pct(bucket.losses, bucket.rows),
            mean(bucket.score_sum, bucket.rows),
            mean_abs(bucket.abs_score_sum, bucket.rows),
            pct(bucket.score_result_agree, bucket.non_draws)
        );
    }
}
