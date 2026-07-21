use std::io::{self, BufRead};
use std::str::FromStr;

use bullet_lib::game::formats::bulletformat::ChessBoard;
use enyo_bullet_spike::enyo_threats;
use serde_json::json;

fn parse_board(fen: &str) -> ChessBoard {
    ChessBoard::from_str(&format!("{fen}|0|0.5"))
        .unwrap_or_else(|err| panic!("failed to parse FEN '{fen}': {err}"))
}

fn values(features: &enyo_threats::ActiveFeatures) -> Vec<usize> {
    (0..features.len()).map(|idx| features.get(idx)).collect()
}

fn dump(fen: &str, reckless: bool) {
    let board = parse_board(fen);
    let features = if reckless {
        enyo_threats::reckless_active_features(&board)
    } else {
        enyo_threats::active_features(&board)
    };
    println!(
        "{}",
        json!({
            "fen": fen,
            "stm": values(&features[0]),
            "ntm": values(&features[1]),
        })
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let reckless = args.first().is_some_and(|arg| arg == "--reckless");
    let args = if reckless { &args[1..] } else { &args[..] };
    if args.is_empty() {
        for line in io::stdin().lock().lines() {
            let fen = line.expect("stdin read failed");
            let fen = fen.trim();
            if !fen.is_empty() {
                dump(fen, reckless);
            }
        }
    } else {
        for fen in args {
            dump(&fen, reckless);
        }
    }
}
