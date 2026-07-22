use std::sync::OnceLock;

use bullet_lib::game::formats::bulletformat::ChessBoard;

pub const DIMENSIONS: usize = 60_720;
pub const RECKLESS_DIMENSIONS: usize = 66_864;
pub const MAX_ACTIVE: usize = 128;

const SQUARES: usize = 64;
const PIECES: usize = 16;
const NONE: u8 = 0xff;

const VALID_TARGETS: [usize; PIECES] = [0, 6, 10, 8, 8, 10, 0, 0, 0, 6, 10, 8, 8, 10, 0, 0];

const TARGET_MAP: [[i32; 6]; 6] = [
    [0, 1, -1, 2, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [-1, -1, -1, -1, -1, -1],
];

const ALL_PIECES: [usize; 12] = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14];

#[derive(Clone, Copy)]
struct HelperOffset {
    piece_span: usize,
    global: usize,
}

struct IndexTables {
    helper: [HelperOffset; PIECES],
    offsets: [[usize; SQUARES]; PIECES],
    target_offsets: [[[usize; 2]; PIECES]; PIECES],
    attack_offsets: [[[u8; SQUARES]; SQUARES]; PIECES],
}

#[derive(Clone)]
pub struct ActiveFeatures {
    values: [usize; MAX_ACTIVE],
    size: usize,
}

impl ActiveFeatures {
    fn new() -> Self {
        Self {
            values: [0; MAX_ACTIVE],
            size: 0,
        }
    }

    fn push(&mut self, index: usize) {
        assert!(self.size < MAX_ACTIVE, "too many FullThreats features");
        self.values[self.size] = index;
        self.size += 1;
    }

    fn sort(&mut self) {
        self.values[..self.size].sort_unstable();
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn get(&self, index: usize) -> usize {
        self.values[index]
    }
}

struct BoardView {
    occupied: u64,
    color: [u64; 2],
    pieces: [[u64; 6]; 2],
    piece_at: [u8; SQUARES],
}

impl BoardView {
    fn empty() -> Self {
        Self {
            occupied: 0,
            color: [0; 2],
            pieces: [[0; 6]; 2],
            piece_at: [NONE; SQUARES],
        }
    }

    fn from_bullet(pos: &ChessBoard) -> Self {
        let mut board = Self::empty();
        for (piece, square) in (*pos).into_iter() {
            let color = usize::from(piece & 8 != 0);
            let piece_type = usize::from(piece & 7);
            let square = usize::from(square);
            let bit = 1_u64 << square;
            board.occupied |= bit;
            board.color[color] |= bit;
            board.pieces[color][piece_type] |= bit;
            board.piece_at[square] = piece;
        }
        board
    }
}

fn tables(reckless: bool) -> &'static IndexTables {
    static TABLES: OnceLock<IndexTables> = OnceLock::new();
    static RECKLESS_TABLES: OnceLock<IndexTables> = OnceLock::new();
    if reckless {
        RECKLESS_TABLES.get_or_init(|| IndexTables::new(true))
    } else {
        TABLES.get_or_init(|| IndexTables::new(false))
    }
}

impl IndexTables {
    fn new(reckless: bool) -> Self {
        let dimensions = if reckless {
            RECKLESS_DIMENSIONS
        } else {
            DIMENSIONS
        };
        let mut tables = Self {
            helper: [HelperOffset {
                piece_span: 0,
                global: 0,
            }; PIECES],
            offsets: [[0; SQUARES]; PIECES],
            target_offsets: [[[dimensions; 2]; PIECES]; PIECES],
            attack_offsets: [[[0; SQUARES]; SQUARES]; PIECES],
        };

        let mut global = 0;
        for &piece in &ALL_PIECES {
            let mut piece_span = 0;
            for square in 0..SQUARES {
                tables.offsets[piece][square] = piece_span;
                if piece_type(piece) != 1 || (8..56).contains(&square) {
                    piece_span += pseudo_attacks(piece, square, reckless).count_ones() as usize;
                }
            }
            tables.helper[piece] = HelperOffset { piece_span, global };
            let valid_targets = if reckless && piece_type(piece) == 6 {
                8
            } else {
                VALID_TARGETS[piece]
            };
            global += valid_targets * piece_span;

            for from in 0..SQUARES {
                let attacks = pseudo_attacks(piece, from, reckless);
                for to in 0..SQUARES {
                    let below = if to == 0 { 0 } else { (1_u64 << to) - 1 };
                    tables.attack_offsets[piece][from][to] = (attacks & below).count_ones() as u8;
                }
            }
        }
        assert_eq!(global, dimensions);

        for &attacker in &ALL_PIECES {
            for &attacked in &ALL_PIECES {
                let attacker_type = piece_type(attacker);
                let attacked_type = piece_type(attacked);
                let mapped_target = if reckless && attacker_type == 6 {
                    [0, 1, 2, 3, -1, -1][attacked_type - 1]
                } else {
                    TARGET_MAP[attacker_type - 1][attacked_type - 1]
                };
                let excluded = mapped_target < 0;
                let enemy = (attacker ^ attacked) == 8;
                let same_type_excluded =
                    attacker_type == attacked_type && (enemy || attacker_type != 1);
                let base = if excluded {
                    dimensions
                } else {
                    let helper = tables.helper[attacker];
                    let valid_targets = if reckless && attacker_type == 6 {
                        8
                    } else {
                        VALID_TARGETS[attacker]
                    };
                    helper.global
                        + (piece_color(attacked) * (valid_targets / 2) + mapped_target as usize)
                            * helper.piece_span
                };
                tables.target_offsets[attacker][attacked][0] = base;
                tables.target_offsets[attacker][attacked][1] = if excluded || same_type_excluded {
                    dimensions
                } else {
                    base
                };
            }
        }

        tables
    }
}

fn on_board(file: i32, rank: i32) -> bool {
    (0..8).contains(&file) && (0..8).contains(&rank)
}

fn piece_type(piece: usize) -> usize {
    piece & 7
}

fn piece_color(piece: usize) -> usize {
    piece >> 3
}

fn stockfish_piece(piece: u8) -> usize {
    usize::from((piece & 7) + 1 + (piece & 8))
}

fn leaper_attacks(piece_type: usize, square: usize) -> u64 {
    const KNIGHT: [(i32, i32); 8] = [
        (1, 2),
        (2, 1),
        (2, -1),
        (1, -2),
        (-1, -2),
        (-2, -1),
        (-2, 1),
        (-1, 2),
    ];
    const KING: [(i32, i32); 8] = [
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
    ];
    let deltas = if piece_type == 2 { &KNIGHT } else { &KING };
    let file = (square % 8) as i32;
    let rank = (square / 8) as i32;
    let mut attacks = 0_u64;
    for &(df, dr) in deltas {
        let to_file = file + df;
        let to_rank = rank + dr;
        if on_board(to_file, to_rank) {
            attacks |= 1_u64 << (to_rank * 8 + to_file);
        }
    }
    attacks
}

fn slider_attacks(piece_type: usize, square: usize, occupied: u64) -> u64 {
    const BISHOP: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, -1), (-1, 1)];
    const ROOK: [(i32, i32); 4] = [(1, 0), (0, -1), (-1, 0), (0, 1)];
    let file = (square % 8) as i32;
    let rank = (square / 8) as i32;
    let mut attacks = 0_u64;
    let mut add_rays = |deltas: &[(i32, i32)]| {
        for &(df, dr) in deltas {
            let mut to_file = file + df;
            let mut to_rank = rank + dr;
            while on_board(to_file, to_rank) {
                let target = (to_rank * 8 + to_file) as usize;
                let bit = 1_u64 << target;
                attacks |= bit;
                if occupied & bit != 0 {
                    break;
                }
                to_file += df;
                to_rank += dr;
            }
        }
    };
    if piece_type == 3 || piece_type == 5 {
        add_rays(&BISHOP);
    }
    if piece_type == 4 || piece_type == 5 {
        add_rays(&ROOK);
    }
    attacks
}

fn pawn_push_or_attacks(color: usize, square: usize) -> u64 {
    let file = (square % 8) as i32;
    let rank = (square / 8) as i32;
    let rank_delta = if color == 0 { 1 } else { -1 };
    let mut attacks = 0_u64;
    for file_delta in [-1, 0, 1] {
        let to_file = file + file_delta;
        let to_rank = rank + rank_delta;
        if on_board(to_file, to_rank) {
            attacks |= 1_u64 << (to_rank * 8 + to_file);
        }
    }
    attacks
}

fn pawn_attacks(color: usize, square: usize) -> u64 {
    let file = (square % 8) as i32;
    let rank = (square / 8) as i32;
    let rank_delta = if color == 0 { 1 } else { -1 };
    let mut attacks = 0_u64;
    for file_delta in [-1, 1] {
        let to_file = file + file_delta;
        let to_rank = rank + rank_delta;
        if on_board(to_file, to_rank) {
            attacks |= 1_u64 << (to_rank * 8 + to_file);
        }
    }
    attacks
}

fn pseudo_attacks(piece: usize, square: usize, reckless: bool) -> u64 {
    match piece_type(piece) {
        1 if reckless => pawn_attacks(piece_color(piece), square),
        1 => pawn_push_or_attacks(piece_color(piece), square),
        2 | 6 => leaper_attacks(piece_type(piece), square),
        _ => slider_attacks(piece_type(piece), square, 0),
    }
}

fn make_index(
    perspective: usize,
    attacker: usize,
    from: usize,
    to: usize,
    attacked: usize,
    king_square: usize,
    reckless: bool,
) -> usize {
    let orientation = (if king_square % 8 < 4 { 0 } else { 7 }) ^ (56 * perspective);
    let oriented_from = from ^ orientation;
    let oriented_to = to ^ orientation;
    let color_swap = 8 * perspective;
    let oriented_attacker = attacker ^ color_swap;
    let oriented_attacked = attacked ^ color_swap;
    let tables = tables(reckless);
    let target_offset = tables.target_offsets[oriented_attacker][oriented_attacked]
        [usize::from(oriented_from < oriented_to)];
    let dimensions = if reckless {
        RECKLESS_DIMENSIONS
    } else {
        DIMENSIONS
    };
    if target_offset >= dimensions {
        return dimensions;
    }
    target_offset
        + tables.offsets[oriented_attacker][oriented_from]
        + usize::from(tables.attack_offsets[oriented_attacker][oriented_from][oriented_to])
}

pub fn active_features(pos: &ChessBoard) -> [ActiveFeatures; 2] {
    let board = BoardView::from_bullet(pos);
    let both = |piece_type: usize| board.pieces[0][piece_type] | board.pieces[1][piece_type];
    let pawn_targets = both(0) | both(1) | both(3);
    let minor_slider_targets = pawn_targets | both(2);
    let queen_targets = minor_slider_targets | both(4);
    let king_square = [
        board.pieces[0][5].trailing_zeros() as usize,
        board.pieces[1][5].trailing_zeros() as usize,
    ];

    let mut features = [ActiveFeatures::new(), ActiveFeatures::new()];
    let mut emit = |attacker: usize, from: usize, to: usize| {
        let attacked_piece = board.piece_at[to];
        if attacked_piece == NONE {
            return;
        }
        let attacked = stockfish_piece(attacked_piece);
        for perspective in 0..2 {
            let index = make_index(
                perspective,
                attacker,
                from,
                to,
                attacked,
                king_square[perspective],
                false,
            );
            if index < DIMENSIONS {
                features[perspective].push(index);
            }
        }
    };

    for color in 0..2 {
        let rank_delta = if color == 0 { 1 } else { -1 };
        let pawn = 1 + color * 8;
        let mut pawns = board.pieces[color][0];
        while pawns != 0 {
            let from = pawns.trailing_zeros() as usize;
            pawns &= pawns - 1;
            let file = (from % 8) as i32;
            let target_rank = (from / 8) as i32 + rank_delta;

            for file_delta in [-1, 1] {
                let target_file = file + file_delta;
                if on_board(target_file, target_rank) {
                    let to = (target_rank * 8 + target_file) as usize;
                    if pawn_targets & (1_u64 << to) != 0 {
                        emit(pawn, from, to);
                    }
                }
            }

            if on_board(file, target_rank) {
                let to = (target_rank * 8 + file) as usize;
                if board.piece_at[to] != NONE && board.piece_at[to] & 7 == 0 {
                    emit(pawn, from, to);
                }
            }
        }

        for piece_type in 1..5 {
            let attacker = piece_type + 1 + color * 8;
            let targets = if piece_type == 1 || piece_type == 4 {
                queen_targets
            } else {
                minor_slider_targets
            };
            let mut attackers = board.pieces[color][piece_type];
            while attackers != 0 {
                let from = attackers.trailing_zeros() as usize;
                attackers &= attackers - 1;
                let attacks = match piece_type {
                    1 => leaper_attacks(2, from),
                    2 => slider_attacks(3, from, board.occupied),
                    3 => slider_attacks(4, from, board.occupied),
                    _ => {
                        slider_attacks(3, from, board.occupied)
                            | slider_attacks(4, from, board.occupied)
                    }
                } & targets;
                let mut hits = attacks;
                while hits != 0 {
                    let to = hits.trailing_zeros() as usize;
                    hits &= hits - 1;
                    emit(attacker, from, to);
                }
            }
        }
    }

    features[0].sort();
    features[1].sort();
    features
}

pub fn reckless_active_features(pos: &ChessBoard) -> [ActiveFeatures; 2] {
    let board = BoardView::from_bullet(pos);
    let king_square = [
        board.pieces[0][5].trailing_zeros() as usize,
        board.pieces[1][5].trailing_zeros() as usize,
    ];
    let mut features = [ActiveFeatures::new(), ActiveFeatures::new()];

    for color in 0..2 {
        for piece_type_index in 0..6 {
            let attacker = piece_type_index + 1 + color * 8;
            let mut attackers = board.pieces[color][piece_type_index];
            while attackers != 0 {
                let from = attackers.trailing_zeros() as usize;
                attackers &= attackers - 1;
                let mut hits = match piece_type_index {
                    0 => pawn_attacks(color, from),
                    1 => leaper_attacks(2, from),
                    2 => slider_attacks(3, from, board.occupied),
                    3 => slider_attacks(4, from, board.occupied),
                    4 => {
                        slider_attacks(3, from, board.occupied)
                            | slider_attacks(4, from, board.occupied)
                    }
                    _ => leaper_attacks(6, from),
                } & board.occupied;
                while hits != 0 {
                    let to = hits.trailing_zeros() as usize;
                    hits &= hits - 1;
                    let attacked = stockfish_piece(board.piece_at[to]);
                    for perspective in 0..2 {
                        let index = make_index(
                            perspective,
                            attacker,
                            from,
                            to,
                            attacked,
                            king_square[perspective],
                            true,
                        );
                        if index < RECKLESS_DIMENSIONS {
                            features[perspective].push(index);
                        }
                    }
                }
            }
        }
    }
    features[0].sort();
    features[1].sort();
    features
}

/// Activates only the first occupied square revealed behind the first blocker
/// on each bishop, rook, or queen ray. Ordinary direct attacks are deliberately
/// excluded so this feature family can be tested independently of FullThreats
/// and Reckless threats. Indices use the established Reckless interaction
/// FullThreats interaction schema for trainer/runtime parity.
pub fn slider_xray_active_features(pos: &ChessBoard) -> [ActiveFeatures; 2] {
    let board = BoardView::from_bullet(pos);
    let king_square = [
        board.pieces[0][5].trailing_zeros() as usize,
        board.pieces[1][5].trailing_zeros() as usize,
    ];
    let mut features = [ActiveFeatures::new(), ActiveFeatures::new()];

    for color in 0..2 {
        for piece_type_index in 2..=4 {
            let attacker = piece_type_index + 1 + color * 8;
            let mut attackers = board.pieces[color][piece_type_index];
            while attackers != 0 {
                let from = attackers.trailing_zeros() as usize;
                attackers &= attackers - 1;
                let attacks = match piece_type_index {
                    2 => slider_attacks(3, from, board.occupied),
                    3 => slider_attacks(4, from, board.occupied),
                    _ => {
                        slider_attacks(3, from, board.occupied)
                            | slider_attacks(4, from, board.occupied)
                    }
                };
                let mut blockers = attacks & board.occupied;
                while blockers != 0 {
                    let blocker = blockers.trailing_zeros() as usize;
                    blockers &= blockers - 1;
                    let occupied_without = board.occupied & !(1_u64 << blocker);
                    let attacks_through = match piece_type_index {
                        2 => slider_attacks(3, from, occupied_without),
                        3 => slider_attacks(4, from, occupied_without),
                        _ => {
                            slider_attacks(3, from, occupied_without)
                                | slider_attacks(4, from, occupied_without)
                        }
                    };
                    let mut revealed = attacks_through & !attacks & occupied_without;
                    while revealed != 0 {
                        let to = revealed.trailing_zeros() as usize;
                        revealed &= revealed - 1;
                        let attacked = stockfish_piece(board.piece_at[to]);
                        for perspective in 0..2 {
                            let index = make_index(
                                perspective,
                                attacker,
                                from,
                                to,
                                attacked,
                                king_square[perspective],
                                false,
                            );
                            if index < DIMENSIONS {
                                features[perspective].push(index);
                            }
                        }
                    }
                }
            }
        }
    }
    features[0].sort();
    features[1].sort();
    features
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn slider_xray_excludes_direct_threat_indices() {
        let board = ChessBoard::from_str("r3k2r/8/n7/p7/P7/8/8/R3K2R w KQkq - 0 1|0|0.5")
            .expect("fixture FEN must parse");
        let features = slider_xray_active_features(&board);
        let expected = [
            [11_418, 12_324, 43_462, 46_153].as_slice(),
            [8_739, 11_418, 39_877, 46_153].as_slice(),
        ];
        for perspective in 0..2 {
            assert_eq!(features[perspective].len(), expected[perspective].len());
            for (offset, expected_index) in expected[perspective].iter().enumerate() {
                assert_eq!(features[perspective].get(offset), *expected_index);
            }
        }
    }
}
