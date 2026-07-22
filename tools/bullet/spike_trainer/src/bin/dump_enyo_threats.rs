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

fn dump(fen: &str, mode: &str) {
    let board = parse_board(fen);
    let features = match mode {
        "--reckless" => enyo_threats::reckless_active_features(&board),
        "--slider-xray" => enyo_threats::slider_xray_active_features(&board),
        _ => enyo_threats::active_features(&board),
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
    let mode = args
        .first()
        .filter(|arg| matches!(arg.as_str(), "--reckless" | "--slider-xray"))
        .map_or("", String::as_str);
    let args = if mode.is_empty() {
        &args[..]
    } else {
        &args[1..]
    };
    if args.is_empty() {
        for line in io::stdin().lock().lines() {
            let fen = line.expect("stdin read failed");
            let fen = fen.trim();
            if !fen.is_empty() {
                dump(fen, mode);
            }
        }
    } else {
        for fen in args {
            dump(&fen, mode);
        }
    }
}
