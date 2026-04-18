use std::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

const INITIAL_PIECES: [Piece; 64] = [
    Piece::WhiteRook,
    Piece::WhiteKnight,
    Piece::WhiteBishop,
    Piece::WhiteQueen,
    Piece::WhiteKing,
    Piece::WhiteBishop,
    Piece::WhiteKnight,
    Piece::WhiteRook,
    Piece::WhitePawn,
    Piece::WhitePawn,
    Piece::WhitePawn,
    Piece::WhitePawn,
    Piece::WhitePawn,
    Piece::WhitePawn,
    Piece::WhitePawn,
    Piece::WhitePawn,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::Empty,
    Piece::BlackPawn,
    Piece::BlackPawn,
    Piece::BlackPawn,
    Piece::BlackPawn,
    Piece::BlackPawn,
    Piece::BlackPawn,
    Piece::BlackPawn,
    Piece::BlackPawn,
    Piece::BlackRook,
    Piece::BlackKnight,
    Piece::BlackBishop,
    Piece::BlackQueen,
    Piece::BlackKing,
    Piece::BlackBishop,
    Piece::BlackKnight,
    Piece::BlackRook,
];

pub const SEARCH_TIME_BUDGET_MS: u32 = 3_000;
const CHECKMATE_SCORE: i32 = 100_000;
const MAX_SEARCH_DEPTH: usize = 64;
const MOVE_FLAG_SHIFT: u32 = 12;
const FILE_MASKS: [Bitboard; 8] = [
    0x0101010101010101,
    0x0202020202020202,
    0x0404040404040404,
    0x0808080808080808,
    0x1010101010101010,
    0x2020202020202020,
    0x4040404040404040,
    0x8080808080808080,
];
const RANK_MASKS: [Bitboard; 8] = [
    0x00000000000000FF,
    0x000000000000FF00,
    0x0000000000FF0000,
    0x00000000FF000000,
    0x000000FF00000000,
    0x0000FF0000000000,
    0x00FF000000000000,
    0xFF00000000000000,
];
const FILE_ABB: Bitboard = FILE_MASKS[0];
const FILE_HBB: Bitboard = FILE_MASKS[7];
const DOUBLE_PUSH_MASK: [Bitboard; 2] = [RANK_MASKS[2], RANK_MASKS[5]];
const TT_BITS: usize = 18;
const TT_SIZE: usize = 1 << TT_BITS;
const TIME_CHECK_INTERVAL: u64 = 1 << 10;
const MAX_GAME_PHASE: i32 = 24;
const ZOBRIST_SEED_INC: u64 = 0x9E37_79B9_7F4A_7C15;
const WHITE_KINGSIDE_CASTLE: u8 = 1 << 0;
const WHITE_QUEENSIDE_CASTLE: u8 = 1 << 1;
const BLACK_KINGSIDE_CASTLE: u8 = 1 << 2;
const BLACK_QUEENSIDE_CASTLE: u8 = 1 << 3;
const ALL_CASTLING_RIGHTS: u8 =
    WHITE_KINGSIDE_CASTLE | WHITE_QUEENSIDE_CASTLE | BLACK_KINGSIDE_CASTLE | BLACK_QUEENSIDE_CASTLE;
const WHITE_KING_START: Square = 4;
const WHITE_KINGSIDE_ROOK_START: Square = 7;
const WHITE_QUEENSIDE_ROOK_START: Square = 0;
const BLACK_KING_START: Square = 60;
const BLACK_KINGSIDE_ROOK_START: Square = 63;
const BLACK_QUEENSIDE_ROOK_START: Square = 56;
const KNIGHT_MOBILITY_BONUS: i32 = 4;
const BISHOP_MOBILITY_BONUS: i32 = 4;
const ROOK_MOBILITY_BONUS: i32 = 2;
const QUEEN_MOBILITY_BONUS: i32 = 1;
const PRIMARY_KILLER_BONUS: i32 = 7_000;
const SECONDARY_KILLER_BONUS: i32 = 6_000;
const MAX_HISTORY_BONUS: i32 = 4_000;

#[cfg(not(target_arch = "wasm32"))]
type SearchDeadline = Instant;

#[cfg(target_arch = "wasm32")]
type SearchDeadline = f64;

type Bitboard = u64;
pub type Square = u32;
pub type Move = u32;

#[repr(u8)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Color {
    White,
    Black,
}

impl Color {
    fn opposite(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

#[repr(u8)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum PieceKind {
    Empty,
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

#[repr(i32)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Piece {
    Empty,
    WhitePawn,
    BlackPawn,
    WhiteKnight,
    BlackKnight,
    WhiteBishop,
    BlackBishop,
    WhiteRook,
    BlackRook,
    WhiteQueen,
    BlackQueen,
    WhiteKing,
    BlackKing,
}

impl Piece {
    fn kind(self) -> PieceKind {
        match self {
            Piece::Empty => PieceKind::Empty,
            Piece::WhitePawn | Piece::BlackPawn => PieceKind::Pawn,
            Piece::WhiteKnight | Piece::BlackKnight => PieceKind::Knight,
            Piece::WhiteBishop | Piece::BlackBishop => PieceKind::Bishop,
            Piece::WhiteRook | Piece::BlackRook => PieceKind::Rook,
            Piece::WhiteQueen | Piece::BlackQueen => PieceKind::Queen,
            Piece::WhiteKing | Piece::BlackKing => PieceKind::King,
        }
    }

    fn color(self) -> Color {
        match self {
            Piece::WhitePawn
            | Piece::WhiteKnight
            | Piece::WhiteBishop
            | Piece::WhiteRook
            | Piece::WhiteQueen
            | Piece::WhiteKing => Color::White,
            Piece::BlackPawn
            | Piece::BlackKnight
            | Piece::BlackBishop
            | Piece::BlackRook
            | Piece::BlackQueen
            | Piece::BlackKing => Color::Black,
            Piece::Empty => panic!("empty piece has no color"),
        }
    }

    fn from_kind_color(kind: PieceKind, color: Color) -> Piece {
        match (kind, color) {
            (PieceKind::Empty, _) => Piece::Empty,
            (PieceKind::Pawn, Color::White) => Piece::WhitePawn,
            (PieceKind::Pawn, Color::Black) => Piece::BlackPawn,
            (PieceKind::Knight, Color::White) => Piece::WhiteKnight,
            (PieceKind::Knight, Color::Black) => Piece::BlackKnight,
            (PieceKind::Bishop, Color::White) => Piece::WhiteBishop,
            (PieceKind::Bishop, Color::Black) => Piece::BlackBishop,
            (PieceKind::Rook, Color::White) => Piece::WhiteRook,
            (PieceKind::Rook, Color::Black) => Piece::BlackRook,
            (PieceKind::Queen, Color::White) => Piece::WhiteQueen,
            (PieceKind::Queen, Color::Black) => Piece::BlackQueen,
            (PieceKind::King, Color::White) => Piece::WhiteKing,
            (PieceKind::King, Color::Black) => Piece::BlackKing,
        }
    }
}

#[repr(i32)]
#[derive(Copy, Clone)]
enum Direction {
    North = 8,
    East = 1,
    South = -8,
    West = -1,
    NorthEast = 9,
    SouthEast = -7,
    SouthWest = -9,
    NorthWest = 7,
}

const BISHOP_DIRECTIONS: [Direction; 4] = [
    Direction::NorthEast,
    Direction::NorthWest,
    Direction::SouthEast,
    Direction::SouthWest,
];
const ROOK_DIRECTIONS: [Direction; 4] = [
    Direction::North,
    Direction::East,
    Direction::South,
    Direction::West,
];
const KING_DIRECTIONS: [Direction; 8] = [
    Direction::North,
    Direction::NorthEast,
    Direction::East,
    Direction::SouthEast,
    Direction::South,
    Direction::SouthWest,
    Direction::West,
    Direction::NorthWest,
];

#[derive(Copy, Clone)]
pub struct Position {
    positions: [Bitboard; 7],
    colors: [Bitboard; 2],
    by_piece: [Piece; 64],
    side_to_move: Color,
    castling_rights: u8,
    en_passant_square: Option<Square>,
    halfmove_clock: u16,
    hash: u64,
}

#[derive(Debug, Copy, Clone, Default)]
pub struct SearchStats {
    pub nodes: u64,
    pub score: i32,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum GameResult {
    Ongoing,
    Draw,
    WhiteWin,
    BlackWin,
}

#[derive(Copy, Clone)]
struct Undo {
    moving_piece: Piece,
    placed_piece: Piece,
    captured_piece: Piece,
    captured_square: Square,
    previous_castling_rights: u8,
    previous_en_passant_square: Option<Square>,
    previous_halfmove_clock: u16,
}

#[derive(Copy, Clone)]
struct Moves {
    moves: [Move; 256],
    num_moves: usize,
}

struct MoveIter<'a> {
    moves: &'a Moves,
    idx: usize,
}

#[derive(Copy, Clone)]
struct SearchWindow {
    alpha: i32,
    beta: i32,
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum Bound {
    Empty,
    Exact,
    Lower,
    Upper,
}

#[derive(Copy, Clone)]
struct TTEntry {
    key: u64,
    depth: u8,
    score: i32,
    best_move: Move,
    bound: Bound,
}

impl Default for TTEntry {
    fn default() -> Self {
        TTEntry {
            key: 0,
            depth: 0,
            score: 0,
            best_move: 0,
            bound: Bound::Empty,
        }
    }
}

struct TranspositionTable {
    entries: Vec<TTEntry>,
    mask: usize,
}

pub struct Searcher {
    tt: TranspositionTable,
    repetition_history: Vec<u64>,
    quiet_history: [[i32; 64]; 64],
    killers: [[Move; 2]; MAX_SEARCH_DEPTH],
}

#[derive(Copy, Clone, Debug)]
enum SearchAborted {
    Timeout,
}

#[repr(u32)]
#[derive(Copy, Clone, Eq, PartialEq)]
enum MoveFlag {
    Quiet,
    DoublePawnPush,
    KingCastle,
    QueenCastle,
    EnPassant,
    PromoteKnight,
    PromoteBishop,
    PromoteRook,
    PromoteQueen,
}

const fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

const fn generate_piece_hashes() -> [[u64; 64]; 13] {
    let mut table = [[0; 64]; 13];
    let mut piece = 1;
    let mut seed: u64 = 0x1234_5678_9ABC_DEF0;

    while piece < 13 {
        let mut square = 0;
        while square < 64 {
            seed = seed.wrapping_add(ZOBRIST_SEED_INC);
            table[piece][square] = mix64(seed);
            square += 1;
        }
        piece += 1;
    }

    table
}

const fn generate_castling_hashes() -> [u64; 16] {
    let mut table = [0; 16];
    let mut idx = 0;
    let mut seed: u64 = 0x0BAD_5EED_C0DE_CAFE;

    while idx < 16 {
        seed = seed.wrapping_add(ZOBRIST_SEED_INC);
        table[idx] = mix64(seed);
        idx += 1;
    }

    table
}

const fn generate_en_passant_hashes() -> [u64; 8] {
    let mut table = [0; 8];
    let mut idx = 0;
    let mut seed: u64 = 0xFACE_FEED_1234_5678;

    while idx < 8 {
        seed = seed.wrapping_add(ZOBRIST_SEED_INC);
        table[idx] = mix64(seed);
        idx += 1;
    }

    table
}

const fn generate_halfmove_hashes() -> [u64; 101] {
    let mut table = [0; 101];
    let mut idx = 0;
    let mut seed: u64 = 0xBADC_0FFE_EE00_DD11;

    while idx < 101 {
        seed = seed.wrapping_add(ZOBRIST_SEED_INC);
        table[idx] = mix64(seed);
        idx += 1;
    }

    table
}

const ZOBRIST_PIECES: [[u64; 64]; 13] = generate_piece_hashes();
const ZOBRIST_SIDE_TO_MOVE: u64 = mix64(0xCAFEBABE_DEADBEEF);
const ZOBRIST_CASTLING: [u64; 16] = generate_castling_hashes();
const ZOBRIST_EN_PASSANT_FILE: [u64; 8] = generate_en_passant_hashes();
const ZOBRIST_HALFMOVE: [u64; 101] = generate_halfmove_hashes();

fn piece_hash(piece: Piece, square: Square) -> u64 {
    ZOBRIST_PIECES[piece as usize][square as usize]
}

fn piece_value(kind: PieceKind) -> i32 {
    match kind {
        PieceKind::Empty => 0,
        PieceKind::Pawn => 100,
        PieceKind::Knight => 320,
        PieceKind::Bishop => 330,
        PieceKind::Rook => 500,
        PieceKind::Queen => 900,
        PieceKind::King => 20_000,
    }
}

const PAWN_PST: [i32; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, -20, -20, 10, 10, 5, 5, -5, -10, 0, 0, -10, -5, 5, 0, 0, 0,
    20, 20, 0, 0, 0, 5, 5, 10, 25, 25, 10, 5, 5, 10, 10, 20, 30, 30, 20, 10, 10, 50, 50, 50, 50,
    50, 50, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0,
];

const KNIGHT_PST: [i32; 64] = [
    -50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 0, 0, 0, -20, -40, -30, 0, 10, 15, 15, 10,
    0, -30, -30, 5, 15, 20, 20, 15, 5, -30, -30, 0, 15, 20, 20, 15, 0, -30, -30, 5, 10, 15, 15, 10,
    5, -30, -40, -20, 0, 5, 5, 0, -20, -40, -50, -40, -30, -30, -30, -30, -40, -50,
];

const BISHOP_PST: [i32; 64] = [
    -20, -10, -10, -10, -10, -10, -10, -20, -10, 5, 0, 0, 0, 0, 5, -10, -10, 10, 10, 10, 10, 10,
    10, -10, -10, 0, 10, 10, 10, 10, 0, -10, -10, 5, 5, 10, 10, 5, 5, -10, -10, 0, 5, 10, 10, 5, 0,
    -10, -10, 0, 0, 0, 0, 0, 0, -10, -20, -10, -10, -10, -10, -10, -10, -20,
];

const ROOK_PST: [i32; 64] = [
    0, 0, 5, 10, 10, 5, 0, 0, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0,
    0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, 5, 10, 10, 10, 10, 10, 10, 5, 0,
    0, 0, 0, 0, 0, 0, 0,
];

const QUEEN_PST: [i32; 64] = [
    -20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5, -5, 0, 5, 5, 5, 5, 0, -5, -10, 0, 5, 5, 5, 5, 0, -10, -10, 0, 0, 0,
    0, 0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20,
];

const KING_MIDDLEGAME_PST: [i32; 64] = [
    20, 30, 10, 0, 0, 10, 30, 20, 20, 20, 0, 0, 0, 0, 20, 20, -10, -20, -20, -20, -20, -20, -20,
    -10, -20, -30, -30, -40, -40, -30, -30, -20, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40,
    -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50,
    -40, -40, -30,
];

const KING_ENDGAME_PST: [i32; 64] = [
    -50, -40, -30, -20, -20, -30, -40, -50, -30, -20, -10, 0, 0, -10, -20, -30, -30, -10, 20, 30,
    30, 20, -10, -30, -30, -10, 30, 40, 40, 30, -10, -30, -30, -10, 30, 40, 40, 30, -10, -30, -30,
    -10, 20, 30, 30, 20, -10, -30, -30, -30, 0, 0, 0, 0, -30, -30, -50, -30, -30, -30, -30, -30,
    -30, -50,
];

fn destination_square(m: Move) -> Square {
    m & 0x3F
}

fn mirror_square(square: Square) -> Square {
    square ^ 56
}

fn origin_square(m: Move) -> Square {
    (m >> 6) & 0x3F
}

fn move_flag(m: Move) -> MoveFlag {
    match (m >> MOVE_FLAG_SHIFT) & 0xF {
        0 => MoveFlag::Quiet,
        1 => MoveFlag::DoublePawnPush,
        2 => MoveFlag::KingCastle,
        3 => MoveFlag::QueenCastle,
        4 => MoveFlag::EnPassant,
        5 => MoveFlag::PromoteKnight,
        6 => MoveFlag::PromoteBishop,
        7 => MoveFlag::PromoteRook,
        8 => MoveFlag::PromoteQueen,
        _ => unreachable!("invalid move flag"),
    }
}

fn promotion_kind(m: Move) -> Option<PieceKind> {
    match move_flag(m) {
        MoveFlag::PromoteKnight => Some(PieceKind::Knight),
        MoveFlag::PromoteBishop => Some(PieceKind::Bishop),
        MoveFlag::PromoteRook => Some(PieceKind::Rook),
        MoveFlag::PromoteQueen => Some(PieceKind::Queen),
        _ => None,
    }
}

fn create_flagged_move(origin: Square, destination: Square, flag: MoveFlag) -> Move {
    let mut m = destination;
    m |= origin << 6;
    m |= (flag as Move) << MOVE_FLAG_SHIFT;
    m
}

fn set_square(bitboard: &mut Bitboard, square: Square) {
    *bitboard |= 1u64 << square;
}

fn unset_square(bitboard: &mut Bitboard, square: Square) {
    *bitboard &= !(1u64 << square);
}

fn pop_square(board: &mut Bitboard) -> Square {
    let square = board.trailing_zeros();
    *board &= *board - 1;
    square
}

fn to_bb(square: Square) -> Bitboard {
    1u64 << square
}

fn shift(bb: Bitboard, dir: Direction) -> Bitboard {
    match dir {
        Direction::North => bb << 8,
        Direction::South => bb >> 8,
        Direction::East => (bb & !FILE_HBB) << 1,
        Direction::West => (bb & !FILE_ABB) >> 1,
        Direction::NorthEast => (bb & !FILE_HBB) << 9,
        Direction::NorthWest => (bb & !FILE_ABB) << 7,
        Direction::SouthEast => (bb & !FILE_HBB) >> 7,
        Direction::SouthWest => (bb & !FILE_ABB) >> 9,
    }
}

fn pawn_push(color: Color) -> Direction {
    match color {
        Color::White => Direction::North,
        Color::Black => Direction::South,
    }
}

fn pawn_capture_directions(color: Color) -> [Direction; 2] {
    match color {
        Color::White => [Direction::NorthWest, Direction::NorthEast],
        Color::Black => [Direction::SouthWest, Direction::SouthEast],
    }
}

fn kingside_castle_right(color: Color) -> u8 {
    match color {
        Color::White => WHITE_KINGSIDE_CASTLE,
        Color::Black => BLACK_KINGSIDE_CASTLE,
    }
}

fn queenside_castle_right(color: Color) -> u8 {
    match color {
        Color::White => WHITE_QUEENSIDE_CASTLE,
        Color::Black => BLACK_QUEENSIDE_CASTLE,
    }
}

fn knight_attacks(from: Square) -> Bitboard {
    let bb = to_bb(from);
    shift(shift(bb, Direction::North), Direction::NorthEast)
        | shift(shift(bb, Direction::North), Direction::NorthWest)
        | shift(shift(bb, Direction::South), Direction::SouthEast)
        | shift(shift(bb, Direction::South), Direction::SouthWest)
        | shift(shift(bb, Direction::East), Direction::NorthEast)
        | shift(shift(bb, Direction::East), Direction::SouthEast)
        | shift(shift(bb, Direction::West), Direction::NorthWest)
        | shift(shift(bb, Direction::West), Direction::SouthWest)
}

fn king_attacks(from: Square) -> Bitboard {
    let bb = to_bb(from);
    let mut attacks = 0;
    for dir in KING_DIRECTIONS {
        attacks |= shift(bb, dir);
    }
    attacks
}

fn ray_attacks(from: Bitboard, occupied: Bitboard, dir: Direction) -> Bitboard {
    let mut attacks = 0;
    let mut current = from;

    loop {
        current = shift(current, dir);
        if current == 0 {
            break;
        }

        attacks |= current;
        if current & occupied != 0 {
            break;
        }
    }

    attacks
}

fn sliding_attacks(from: Square, occupied: Bitboard, directions: &[Direction]) -> Bitboard {
    let mut attacks = 0;
    let bb = to_bb(from);
    for &dir in directions {
        attacks |= ray_attacks(bb, occupied, dir);
    }
    attacks
}

impl Position {
    pub fn new() -> Position {
        Position::from_board_state(INITIAL_PIECES, Color::White, ALL_CASTLING_RIGHTS, None, 0)
    }

    fn from_board_state(
        by_piece: [Piece; 64],
        side_to_move: Color,
        castling_rights: u8,
        en_passant_square: Option<Square>,
        halfmove_clock: u16,
    ) -> Position {
        let mut position = Position {
            positions: [0; 7],
            colors: [0; 2],
            by_piece,
            side_to_move,
            castling_rights,
            en_passant_square,
            halfmove_clock,
            hash: 0,
        };

        for (i, piece) in position.by_piece.iter().copied().enumerate() {
            let square = i as Square;
            set_square(&mut position.positions[piece.kind() as usize], square);
            if piece != Piece::Empty {
                set_square(&mut position.colors[piece.color() as usize], square);
                position.hash ^= piece_hash(piece, square);
            }
        }

        if side_to_move == Color::Black {
            position.hash ^= ZOBRIST_SIDE_TO_MOVE;
        }
        position.hash ^= ZOBRIST_CASTLING[castling_rights as usize];
        if let Some(square) = en_passant_square {
            position.hash ^= ZOBRIST_EN_PASSANT_FILE[(square % 8) as usize];
        }

        position
    }

    pub fn from_fen(fen: &str) -> Result<Position, String> {
        let mut fields = fen.split_whitespace();
        let board_field = fields
            .next()
            .ok_or_else(|| "missing board in FEN".to_string())?;
        let side_field = fields
            .next()
            .ok_or_else(|| "missing side to move in FEN".to_string())?;
        let castling_field = fields
            .next()
            .ok_or_else(|| "missing castling rights in FEN".to_string())?;
        let en_passant_field = fields
            .next()
            .ok_or_else(|| "missing en passant square in FEN".to_string())?;

        let mut board = [Piece::Empty; 64];
        let ranks: Vec<_> = board_field.split('/').collect();
        if ranks.len() != 8 {
            return Err("board must have 8 ranks".to_string());
        }

        for (fen_rank, rank_str) in ranks.iter().enumerate() {
            let rank = 7 - fen_rank as u32;
            let mut file = 0u32;

            for ch in rank_str.chars() {
                if let Some(empty) = ch.to_digit(10) {
                    if empty == 0 || file + empty > 8 {
                        return Err("invalid empty-square run in FEN".to_string());
                    }
                    file += empty;
                    continue;
                }

                let piece = match ch {
                    'P' => Piece::WhitePawn,
                    'p' => Piece::BlackPawn,
                    'N' => Piece::WhiteKnight,
                    'n' => Piece::BlackKnight,
                    'B' => Piece::WhiteBishop,
                    'b' => Piece::BlackBishop,
                    'R' => Piece::WhiteRook,
                    'r' => Piece::BlackRook,
                    'Q' => Piece::WhiteQueen,
                    'q' => Piece::BlackQueen,
                    'K' => Piece::WhiteKing,
                    'k' => Piece::BlackKing,
                    _ => return Err(format!("invalid piece in FEN: {ch}")),
                };

                if file >= 8 {
                    return Err("rank is too wide in FEN".to_string());
                }

                board[(rank * 8 + file) as usize] = piece;
                file += 1;
            }

            if file != 8 {
                return Err("rank is too short in FEN".to_string());
            }
        }

        let side_to_move = match side_field {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err("invalid side to move in FEN".to_string()),
        };

        let mut castling_rights = 0;
        if castling_field != "-" {
            for ch in castling_field.chars() {
                castling_rights |= match ch {
                    'K' => WHITE_KINGSIDE_CASTLE,
                    'Q' => WHITE_QUEENSIDE_CASTLE,
                    'k' => BLACK_KINGSIDE_CASTLE,
                    'q' => BLACK_QUEENSIDE_CASTLE,
                    _ => return Err(format!("invalid castling right in FEN: {ch}")),
                };
            }
        }

        let en_passant_square = if en_passant_field == "-" {
            None
        } else {
            Some(parse_square_name(en_passant_field)?)
        };
        let halfmove_clock = match fields.next() {
            Some(field) => field
                .parse::<u16>()
                .map_err(|_| "invalid halfmove clock in FEN".to_string())?,
            None => 0,
        };

        if let Some(field) = fields.next() {
            let fullmove_number = field
                .parse::<u32>()
                .map_err(|_| "invalid fullmove number in FEN".to_string())?;
            if fullmove_number == 0 {
                return Err("fullmove number must be positive in FEN".to_string());
            }
        }
        if fields.next().is_some() {
            return Err("too many fields in FEN".to_string());
        }

        Ok(Position::from_board_state(
            board,
            side_to_move,
            castling_rights,
            en_passant_square,
            halfmove_clock,
        ))
    }

    pub fn piece_at(&self, square: Square) -> Piece {
        self.by_piece[square as usize]
    }

    pub fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    pub fn castling_rights_bits(&self) -> u8 {
        self.castling_rights
    }

    pub fn en_passant_square(&self) -> Option<Square> {
        self.en_passant_square
    }

    pub fn halfmove_clock(&self) -> u16 {
        self.halfmove_clock
    }

    pub fn history_hash(&self) -> u64 {
        self.hash
    }

    fn tt_key(&self) -> u64 {
        self.hash ^ ZOBRIST_HALFMOVE[self.halfmove_clock.min(100) as usize]
    }

    fn occupied(&self) -> Bitboard {
        self.colors[Color::White as usize] | self.colors[Color::Black as usize]
    }

    fn pieces(&self, color: Color, kind: PieceKind) -> Bitboard {
        self.colors[color as usize] & self.positions[kind as usize]
    }

    fn king_square(&self, color: Color) -> Option<Square> {
        let kings = self.pieces(color, PieceKind::King);
        if kings == 0 {
            None
        } else {
            Some(kings.trailing_zeros())
        }
    }

    fn set_castling_rights(&mut self, castling_rights: u8) {
        self.hash ^= ZOBRIST_CASTLING[self.castling_rights as usize];
        self.castling_rights = castling_rights;
        self.hash ^= ZOBRIST_CASTLING[self.castling_rights as usize];
    }

    fn set_en_passant_square(&mut self, en_passant_square: Option<Square>) {
        if let Some(square) = self.en_passant_square {
            self.hash ^= ZOBRIST_EN_PASSANT_FILE[(square % 8) as usize];
        }
        self.en_passant_square = en_passant_square;
        if let Some(square) = self.en_passant_square {
            self.hash ^= ZOBRIST_EN_PASSANT_FILE[(square % 8) as usize];
        }
    }

    fn set_halfmove_clock(&mut self, halfmove_clock: u16) {
        self.halfmove_clock = halfmove_clock;
    }

    fn update_castling_rights_after_move(
        &mut self,
        moving_piece: Piece,
        origin: Square,
        captured_piece: Piece,
        captured_square: Square,
    ) {
        let mut castling_rights = self.castling_rights;

        castling_rights &= match (moving_piece, origin) {
            (Piece::WhiteKing, _) => !(WHITE_KINGSIDE_CASTLE | WHITE_QUEENSIDE_CASTLE),
            (Piece::BlackKing, _) => !(BLACK_KINGSIDE_CASTLE | BLACK_QUEENSIDE_CASTLE),
            (Piece::WhiteRook, WHITE_KINGSIDE_ROOK_START) => !WHITE_KINGSIDE_CASTLE,
            (Piece::WhiteRook, WHITE_QUEENSIDE_ROOK_START) => !WHITE_QUEENSIDE_CASTLE,
            (Piece::BlackRook, BLACK_KINGSIDE_ROOK_START) => !BLACK_KINGSIDE_CASTLE,
            (Piece::BlackRook, BLACK_QUEENSIDE_ROOK_START) => !BLACK_QUEENSIDE_CASTLE,
            _ => ALL_CASTLING_RIGHTS,
        };

        castling_rights &= match (captured_piece, captured_square) {
            (Piece::WhiteRook, WHITE_KINGSIDE_ROOK_START) => !WHITE_KINGSIDE_CASTLE,
            (Piece::WhiteRook, WHITE_QUEENSIDE_ROOK_START) => !WHITE_QUEENSIDE_CASTLE,
            (Piece::BlackRook, BLACK_KINGSIDE_ROOK_START) => !BLACK_KINGSIDE_CASTLE,
            (Piece::BlackRook, BLACK_QUEENSIDE_ROOK_START) => !BLACK_QUEENSIDE_CASTLE,
            _ => ALL_CASTLING_RIGHTS,
        };

        self.set_castling_rights(castling_rights);
    }

    fn remove_piece(&mut self, square: Square, piece: Piece) {
        unset_square(&mut self.positions[piece.kind() as usize], square);
        if piece != Piece::Empty {
            unset_square(&mut self.colors[piece.color() as usize], square);
            self.hash ^= piece_hash(piece, square);
        }
        self.by_piece[square as usize] = Piece::Empty;
        set_square(&mut self.positions[PieceKind::Empty as usize], square);
    }

    fn add_piece(&mut self, square: Square, piece: Piece) {
        unset_square(&mut self.positions[PieceKind::Empty as usize], square);
        set_square(&mut self.positions[piece.kind() as usize], square);
        if piece != Piece::Empty {
            set_square(&mut self.colors[piece.color() as usize], square);
            self.hash ^= piece_hash(piece, square);
        }
        self.by_piece[square as usize] = piece;
    }

    fn make_move_unchecked(&mut self, m: Move) -> Undo {
        let origin = origin_square(m);
        let destination = destination_square(m);
        let flag = move_flag(m);
        let moving_piece = self.by_piece[origin as usize];
        let captured_square = if flag == MoveFlag::EnPassant {
            match self.side_to_move {
                Color::White => destination - 8,
                Color::Black => destination + 8,
            }
        } else {
            destination
        };
        let captured_piece = self.by_piece[captured_square as usize];
        let placed_piece = promotion_kind(m)
            .map(|kind| Piece::from_kind_color(kind, self.side_to_move))
            .unwrap_or_else(|| {
                let destination_rank = destination / 8;
                if moving_piece.kind() == PieceKind::Pawn
                    && (destination_rank == 0 || destination_rank == 7)
                {
                    Piece::from_kind_color(PieceKind::Queen, self.side_to_move)
                } else {
                    moving_piece
                }
            });
        let previous_castling_rights = self.castling_rights;
        let previous_en_passant_square = self.en_passant_square;
        let previous_halfmove_clock = self.halfmove_clock;

        self.set_en_passant_square(None);
        self.set_halfmove_clock(
            if moving_piece.kind() == PieceKind::Pawn || captured_piece != Piece::Empty {
                0
            } else {
                previous_halfmove_clock.saturating_add(1)
            },
        );

        self.remove_piece(origin, moving_piece);
        if captured_piece != Piece::Empty {
            self.remove_piece(captured_square, captured_piece);
        }
        self.add_piece(destination, placed_piece);

        match flag {
            MoveFlag::KingCastle => {
                let (rook_from, rook_to) = match self.side_to_move {
                    Color::White => (WHITE_KINGSIDE_ROOK_START, 5),
                    Color::Black => (BLACK_KINGSIDE_ROOK_START, 61),
                };
                let rook = self.by_piece[rook_from as usize];
                self.remove_piece(rook_from, rook);
                self.add_piece(rook_to, rook);
            }
            MoveFlag::QueenCastle => {
                let (rook_from, rook_to) = match self.side_to_move {
                    Color::White => (WHITE_QUEENSIDE_ROOK_START, 3),
                    Color::Black => (BLACK_QUEENSIDE_ROOK_START, 59),
                };
                let rook = self.by_piece[rook_from as usize];
                self.remove_piece(rook_from, rook);
                self.add_piece(rook_to, rook);
            }
            MoveFlag::DoublePawnPush => {
                let en_passant_square = match self.side_to_move {
                    Color::White => origin + 8,
                    Color::Black => origin - 8,
                };
                let them = self.side_to_move.opposite();
                let enemy_pawns = self.pieces(them, PieceKind::Pawn);
                let [left, right] = pawn_capture_directions(them);
                let en_passant_target = to_bb(en_passant_square);

                if (shift(enemy_pawns, left) | shift(enemy_pawns, right)) & en_passant_target != 0 {
                    self.set_en_passant_square(Some(en_passant_square));
                }
            }
            _ => {}
        }

        self.update_castling_rights_after_move(
            moving_piece,
            origin,
            captured_piece,
            captured_square,
        );

        self.side_to_move = self.side_to_move.opposite();
        self.hash ^= ZOBRIST_SIDE_TO_MOVE;

        Undo {
            moving_piece,
            placed_piece,
            captured_piece,
            captured_square,
            previous_castling_rights,
            previous_en_passant_square,
            previous_halfmove_clock,
        }
    }

    fn undo_move_unchecked(&mut self, m: Move, undo: Undo) {
        let origin = origin_square(m);
        let destination = destination_square(m);
        let flag = move_flag(m);

        self.side_to_move = self.side_to_move.opposite();
        self.hash ^= ZOBRIST_SIDE_TO_MOVE;

        self.set_en_passant_square(undo.previous_en_passant_square);
        self.set_castling_rights(undo.previous_castling_rights);
        self.set_halfmove_clock(undo.previous_halfmove_clock);

        self.remove_piece(destination, undo.placed_piece);
        match flag {
            MoveFlag::KingCastle => {
                let (rook_from, rook_to) = match self.side_to_move {
                    Color::White => (5, WHITE_KINGSIDE_ROOK_START),
                    Color::Black => (61, BLACK_KINGSIDE_ROOK_START),
                };
                let rook = self.by_piece[rook_from as usize];
                self.remove_piece(rook_from, rook);
                self.add_piece(rook_to, rook);
            }
            MoveFlag::QueenCastle => {
                let (rook_from, rook_to) = match self.side_to_move {
                    Color::White => (3, WHITE_QUEENSIDE_ROOK_START),
                    Color::Black => (59, BLACK_QUEENSIDE_ROOK_START),
                };
                let rook = self.by_piece[rook_from as usize];
                self.remove_piece(rook_from, rook);
                self.add_piece(rook_to, rook);
            }
            _ => {}
        }
        self.add_piece(origin, undo.moving_piece);
        if undo.captured_piece != Piece::Empty {
            self.add_piece(undo.captured_square, undo.captured_piece);
        }
    }

    pub fn do_move(&mut self, m: Move) {
        let origin = origin_square(m);
        let destination = destination_square(m);
        let moving_piece = self.by_piece[origin as usize];
        assert!(moving_piece != Piece::Empty, "cannot move an empty square");
        assert_eq!(
            moving_piece.color(),
            self.side_to_move,
            "cannot move the opponent's piece"
        );

        let target_piece = self.by_piece[destination as usize];
        assert!(
            target_piece == Piece::Empty || target_piece.color() != self.side_to_move,
            "cannot capture your own piece"
        );

        let _ = self.make_move_unchecked(m);
        assert!(self.is_consistent());
    }

    fn material(&self, color: Color) -> i32 {
        let color_mask = self.colors[color as usize];
        [
            PieceKind::Pawn,
            PieceKind::Knight,
            PieceKind::Bishop,
            PieceKind::Rook,
            PieceKind::Queen,
            PieceKind::King,
        ]
        .into_iter()
        .map(|kind| {
            (self.positions[kind as usize] & color_mask).count_ones() as i32 * piece_value(kind)
        })
        .sum()
    }

    fn game_phase(&self) -> i32 {
        let mut phase = 0;
        for color in [Color::White, Color::Black] {
            phase += self.pieces(color, PieceKind::Knight).count_ones() as i32;
            phase += self.pieces(color, PieceKind::Bishop).count_ones() as i32;
            phase += self.pieces(color, PieceKind::Rook).count_ones() as i32 * 2;
            phase += self.pieces(color, PieceKind::Queen).count_ones() as i32 * 4;
        }
        phase.min(MAX_GAME_PHASE)
    }

    fn piece_square_index(color: Color, square: Square) -> usize {
        match color {
            Color::White => square as usize,
            Color::Black => mirror_square(square) as usize,
        }
    }

    fn pst_sum(&self, color: Color, kind: PieceKind, table: &[i32; 64]) -> i32 {
        let mut pieces = self.pieces(color, kind);
        let mut score = 0;

        while pieces != 0 {
            let square = pop_square(&mut pieces);
            score += table[Self::piece_square_index(color, square)];
        }

        score
    }

    fn piece_square_score(&self, color: Color, phase: i32) -> i32 {
        let mut score = 0;
        score += self.pst_sum(color, PieceKind::Pawn, &PAWN_PST);
        score += self.pst_sum(color, PieceKind::Knight, &KNIGHT_PST);
        score += self.pst_sum(color, PieceKind::Bishop, &BISHOP_PST);
        score += self.pst_sum(color, PieceKind::Rook, &ROOK_PST);
        score += self.pst_sum(color, PieceKind::Queen, &QUEEN_PST);

        if let Some(square) = self.king_square(color) {
            let index = Self::piece_square_index(color, square);
            let middlegame = KING_MIDDLEGAME_PST[index];
            let endgame = KING_ENDGAME_PST[index];
            score += (middlegame * phase + endgame * (MAX_GAME_PHASE - phase)) / MAX_GAME_PHASE;
        }

        score
    }

    fn development_score(&self, color: Color, phase: i32) -> i32 {
        if phase == 0 {
            return 0;
        }

        let (knight_home, bishop_home, king_start, castled_squares, castling_rights) = match color {
            Color::White => (
                to_bb(1) | to_bb(6),
                to_bb(2) | to_bb(5),
                WHITE_KING_START,
                [6, 2],
                WHITE_KINGSIDE_CASTLE | WHITE_QUEENSIDE_CASTLE,
            ),
            Color::Black => (
                to_bb(57) | to_bb(62),
                to_bb(58) | to_bb(61),
                BLACK_KING_START,
                [62, 58],
                BLACK_KINGSIDE_CASTLE | BLACK_QUEENSIDE_CASTLE,
            ),
        };

        let developed_knights = (self.pieces(color, PieceKind::Knight) & !knight_home)
            .count_ones()
            .min(2) as i32;
        let developed_bishops = (self.pieces(color, PieceKind::Bishop) & !bishop_home)
            .count_ones()
            .min(2) as i32;

        let mut score = developed_knights * 14 + developed_bishops * 12;
        let king_square = self.king_square(color);

        if king_square.is_some_and(|square| castled_squares.contains(&square)) {
            score += 30;
        }
        if king_square == Some(king_start) && self.castling_rights & castling_rights == 0 {
            score -= 20;
        }

        score * phase / MAX_GAME_PHASE
    }

    fn mobility_score_for_kind(&self, color: Color, kind: PieceKind, bonus: i32) -> i32 {
        let own_pieces = self.colors[color as usize];
        let enemy_king = self.pieces(color.opposite(), PieceKind::King);
        let mut pieces = self.pieces(color, kind);
        let mut score = 0;

        while pieces != 0 {
            let from = pop_square(&mut pieces);
            let targets = get_attacks(from, self, kind) & !own_pieces & !enemy_king;
            score += targets.count_ones() as i32 * bonus;
        }

        score
    }

    fn mobility_score(&self, color: Color) -> i32 {
        self.mobility_score_for_kind(color, PieceKind::Knight, KNIGHT_MOBILITY_BONUS)
            + self.mobility_score_for_kind(color, PieceKind::Bishop, BISHOP_MOBILITY_BONUS)
            + self.mobility_score_for_kind(color, PieceKind::Rook, ROOK_MOBILITY_BONUS)
            + self.mobility_score_for_kind(color, PieceKind::Queen, QUEEN_MOBILITY_BONUS)
    }

    fn evaluate_color(&self, color: Color, phase: i32) -> i32 {
        self.material(color)
            + self.piece_square_score(color, phase)
            + self.development_score(color, phase)
            + self.mobility_score(color)
    }

    fn evaluate(&self) -> i32 {
        let phase = self.game_phase();
        self.evaluate_color(self.side_to_move, phase)
            - self.evaluate_color(self.side_to_move.opposite(), phase)
    }

    fn is_consistent(&self) -> bool {
        let occupied = self.occupied();
        for square in 0..64 {
            let piece = self.by_piece[square];
            let bb = 1u64 << square as u64;
            if self.positions[piece.kind() as usize] & bb == 0 {
                return false;
            }
            if piece != Piece::Empty {
                let color = piece.color();
                if self.colors[color as usize] & bb == 0 {
                    return false;
                }
            } else if occupied & bb != 0 {
                return false;
            }
        }
        true
    }

    fn in_check(&self, color: Color) -> bool {
        self.king_square(color)
            .is_some_and(|square| is_square_attacked(self, square, color.opposite()))
    }
}

impl Default for Position {
    fn default() -> Self {
        Self::new()
    }
}

pub fn is_draw_by_rule(position: &Position, position_history: &[u64]) -> bool {
    position.halfmove_clock >= 100 || has_threefold_repetition(position, position_history)
}

fn has_threefold_repetition(position: &Position, position_history: &[u64]) -> bool {
    if position.halfmove_clock < 4 {
        return false;
    }

    let Some(&current_hash) = position_history.last() else {
        return false;
    };
    if current_hash != position.history_hash() {
        return false;
    }

    let mut repetitions = 1;
    let mut plies_back = 2usize;
    let max_plies = position.halfmove_clock as usize;

    while plies_back <= max_plies && plies_back < position_history.len() {
        let idx = position_history.len() - 1 - plies_back;
        if position_history[idx] == current_hash {
            repetitions += 1;
            if repetitions >= 3 {
                return true;
            }
        }
        plies_back += 2;
    }

    false
}

impl Moves {
    fn new() -> Moves {
        Moves {
            moves: [0; 256],
            num_moves: 0,
        }
    }

    fn push(&mut self, m: Move) {
        debug_assert!(self.num_moves < self.moves.len());
        self.moves[self.num_moves] = m;
        self.num_moves += 1;
    }

    fn iter(&self) -> MoveIter<'_> {
        MoveIter {
            moves: self,
            idx: 0,
        }
    }

    fn find(&self, origin: Square, destination: Square) -> Option<Move> {
        self.iter()
            .find(|m| origin_square(*m) == origin && destination_square(*m) == destination)
    }
}

impl<'a> Iterator for MoveIter<'a> {
    type Item = Move;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx == self.moves.num_moves {
            return None;
        }

        let idx = self.idx;
        self.idx += 1;
        Some(self.moves.moves[idx])
    }
}

impl TranspositionTable {
    fn new(size: usize) -> TranspositionTable {
        assert!(size.is_power_of_two());
        TranspositionTable {
            entries: vec![TTEntry::default(); size],
            mask: size - 1,
        }
    }

    fn probe(&self, key: u64) -> Option<TTEntry> {
        let entry = self.entries[key as usize & self.mask];
        if entry.bound != Bound::Empty && entry.key == key {
            Some(entry)
        } else {
            None
        }
    }

    fn store(&mut self, key: u64, depth: usize, score: i32, bound: Bound, best_move: Move) {
        let idx = key as usize & self.mask;
        let current = self.entries[idx];
        if current.bound == Bound::Empty || current.key != key || depth >= current.depth as usize {
            self.entries[idx] = TTEntry {
                key,
                depth: depth as u8,
                score,
                best_move,
                bound,
            };
        }
    }
}

impl Searcher {
    pub fn new() -> Searcher {
        Searcher {
            tt: TranspositionTable::new(TT_SIZE),
            repetition_history: Vec::new(),
            quiet_history: [[0; 64]; 64],
            killers: [[0; 2]; MAX_SEARCH_DEPTH],
        }
    }

    fn prepare_history(&mut self, position: &Position, position_history: &[u64]) {
        self.repetition_history.clear();
        if position_history.is_empty() {
            self.repetition_history.push(position.history_hash());
            return;
        }

        debug_assert_eq!(position_history.last(), Some(&position.history_hash()));
        self.repetition_history.extend_from_slice(position_history);
    }

    fn clear_move_ordering_heuristics(&mut self) {
        self.quiet_history = [[0; 64]; 64];
        self.killers = [[0; 2]; MAX_SEARCH_DEPTH];
    }

    fn quiet_history_score(&self, m: Move) -> i32 {
        self.quiet_history[origin_square(m) as usize][destination_square(m) as usize]
            .min(MAX_HISTORY_BONUS)
    }

    fn note_quiet_cutoff(&mut self, ply: usize, m: Move, depth: usize) {
        let killers = &mut self.killers[ply.min(MAX_SEARCH_DEPTH - 1)];
        if killers[0] != m {
            killers[1] = killers[0];
            killers[0] = m;
        }

        let bonus = (depth * depth) as i32;
        let entry =
            &mut self.quiet_history[origin_square(m) as usize][destination_square(m) as usize];
        *entry = (*entry + bonus).min(32_000);
    }

    fn move_score(&self, position: &Position, m: Move, tt_move: Option<Move>, ply: usize) -> i32 {
        if tt_move.is_some_and(|candidate| candidate == m) {
            return i32::MAX;
        }

        let origin = origin_square(m);
        let moving_piece = position.by_piece[origin as usize];
        let captured_piece = if move_flag(m) == MoveFlag::EnPassant {
            Piece::from_kind_color(PieceKind::Pawn, position.side_to_move.opposite())
        } else {
            position.by_piece[destination_square(m) as usize]
        };

        let mut score = 0;
        if captured_piece != Piece::Empty {
            score += 10_000;
            score += piece_value(captured_piece.kind()) * 16 - piece_value(moving_piece.kind());
        }
        if let Some(kind) = promotion_kind(m) {
            score += 8_000 + piece_value(kind);
        }
        if captured_piece == Piece::Empty && promotion_kind(m).is_none() {
            let killers = self.killers[ply.min(MAX_SEARCH_DEPTH - 1)];
            if killers[0] == m {
                score += PRIMARY_KILLER_BONUS;
            } else if killers[1] == m {
                score += SECONDARY_KILLER_BONUS;
            }
            score += self.quiet_history_score(m);
        }

        score
    }

    fn order_moves(
        &self,
        position: &Position,
        moves: &mut Moves,
        tt_move: Option<Move>,
        ply: usize,
    ) {
        let mut scores = [0; 256];
        for (idx, score) in scores.iter_mut().enumerate().take(moves.num_moves) {
            *score = self.move_score(position, moves.moves[idx], tt_move, ply);
        }

        for i in 1..moves.num_moves {
            let move_to_insert = moves.moves[i];
            let score_to_insert = scores[i];
            let mut j = i;

            while j > 0 && score_to_insert > scores[j - 1] {
                moves.moves[j] = moves.moves[j - 1];
                scores[j] = scores[j - 1];
                j -= 1;
            }

            moves.moves[j] = move_to_insert;
            scores[j] = score_to_insert;
        }
    }

    pub fn make_move(&mut self, position: &mut Position, position_history: &[u64]) -> bool {
        let Some((best_move, _)) = self.best_move_with_time_budget(
            position,
            position_history,
            Duration::from_millis(SEARCH_TIME_BUDGET_MS as u64),
        ) else {
            return false;
        };
        position.do_move(best_move);
        true
    }

    pub fn search_root_with_stats(
        &mut self,
        position: &Position,
        position_history: &[u64],
        depth: usize,
    ) -> Option<SearchStats> {
        self.prepare_history(position, position_history);
        self.clear_move_ordering_heuristics();
        if is_draw_by_rule(position, &self.repetition_history) {
            return Some(SearchStats::default());
        }

        let mut scratch = *position;
        self.search_root(&mut scratch, depth, None)
            .ok()
            .flatten()
            .map(|(_, stats)| stats)
    }

    pub fn best_move_with_time_budget(
        &mut self,
        position: &Position,
        position_history: &[u64],
        time_budget: Duration,
    ) -> Option<(Move, SearchStats)> {
        self.prepare_history(position, position_history);
        self.clear_move_ordering_heuristics();
        if is_draw_by_rule(position, &self.repetition_history) {
            return None;
        }

        let mut scratch = *position;
        let legal_moves = generate_legal_moves(&mut scratch);
        match legal_moves.num_moves {
            0 => return None,
            1 => return Some((legal_moves.moves[0], SearchStats { nodes: 0, score: 0 })),
            _ => {}
        }

        let deadline = deadline_after(time_budget);
        let mut best_completed = None;

        for depth in 1..=MAX_SEARCH_DEPTH {
            if deadline_reached(deadline) {
                break;
            }

            let mut scratch = *position;
            match self.search_root(&mut scratch, depth, Some(deadline)) {
                Ok(Some(result)) => best_completed = Some(result),
                Ok(None) => return None,
                Err(SearchAborted::Timeout) => break,
            }
        }

        best_completed
    }

    fn search_root(
        &mut self,
        position: &mut Position,
        depth: usize,
        deadline: Option<SearchDeadline>,
    ) -> Result<Option<(Move, SearchStats)>, SearchAborted> {
        let mut moves = generate_legal_moves(position);
        if moves.num_moves == 0 {
            return Ok(None);
        }

        let tt_move = self
            .tt
            .probe(position.tt_key())
            .map(|entry| entry.best_move);
        self.order_moves(position, &mut moves, tt_move, 0);

        let mut stats = SearchStats::default();
        let mut best_move = moves.moves[0];
        let mut best_score = -CHECKMATE_SCORE;
        let mut alpha = -CHECKMATE_SCORE;
        let beta = CHECKMATE_SCORE;

        for m in moves.iter() {
            if search_timed_out(deadline, stats.nodes) {
                return Err(SearchAborted::Timeout);
            }
            let undo = position.make_move_unchecked(m);
            self.repetition_history.push(position.history_hash());
            let value = self.negamax(
                position,
                depth.saturating_sub(1),
                SearchWindow {
                    alpha: -beta,
                    beta: -alpha,
                },
                &mut stats,
                deadline,
                1,
            );
            self.repetition_history.pop();
            position.undo_move_unchecked(m, undo);
            let value = -value?;

            if value > best_score {
                best_score = value;
                best_move = m;
            }

            if value > alpha {
                alpha = value;
            }
        }

        self.tt.store(
            position.tt_key(),
            depth,
            best_score,
            Bound::Exact,
            best_move,
        );

        stats.score = best_score;
        Ok(Some((best_move, stats)))
    }

    fn negamax(
        &mut self,
        position: &mut Position,
        depth: usize,
        window: SearchWindow,
        stats: &mut SearchStats,
        deadline: Option<SearchDeadline>,
        ply: usize,
    ) -> Result<i32, SearchAborted> {
        let mut alpha = window.alpha;
        let beta = window.beta;
        if depth == 0 {
            return self.quiescence(position, SearchWindow { alpha, beta }, stats, deadline, ply);
        }

        stats.nodes += 1;
        if search_timed_out(deadline, stats.nodes) {
            return Err(SearchAborted::Timeout);
        }
        if is_draw_by_rule(position, &self.repetition_history) {
            return Ok(0);
        }

        let original_alpha = alpha;
        let tt_entry = self.tt.probe(position.tt_key());
        if let Some(entry) = tt_entry {
            if entry.depth as usize >= depth {
                match entry.bound {
                    Bound::Exact => return Ok(entry.score),
                    Bound::Lower => {
                        if entry.score >= beta {
                            return Ok(entry.score);
                        }
                        alpha = alpha.max(entry.score);
                    }
                    Bound::Upper => {
                        if entry.score <= alpha {
                            return Ok(entry.score);
                        }
                    }
                    Bound::Empty => {}
                }
            }
        }

        let mut moves = generate_legal_moves(position);
        if moves.num_moves == 0 {
            return Ok(if position.in_check(position.side_to_move) {
                -CHECKMATE_SCORE + depth as i32
            } else {
                0
            });
        }

        self.order_moves(
            position,
            &mut moves,
            tt_entry.map(|entry| entry.best_move),
            ply,
        );

        let mut best_value = -CHECKMATE_SCORE;
        let mut best_move = 0;

        for m in moves.iter() {
            let quiet_move = !is_tactical_move(position, m);
            let undo = position.make_move_unchecked(m);
            self.repetition_history.push(position.history_hash());
            let value = self.negamax(
                position,
                depth - 1,
                SearchWindow {
                    alpha: -beta,
                    beta: -alpha,
                },
                stats,
                deadline,
                ply + 1,
            );
            self.repetition_history.pop();
            position.undo_move_unchecked(m, undo);
            let value = -value?;

            if value > best_value {
                best_value = value;
                best_move = m;
            }
            if value > alpha {
                alpha = value;
            }
            if alpha >= beta {
                if quiet_move {
                    self.note_quiet_cutoff(ply, m, depth);
                }
                break;
            }
        }

        let bound = if best_value <= original_alpha {
            Bound::Upper
        } else if best_value >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };

        self.tt
            .store(position.tt_key(), depth, best_value, bound, best_move);
        Ok(best_value)
    }

    fn quiescence(
        &mut self,
        position: &mut Position,
        window: SearchWindow,
        stats: &mut SearchStats,
        deadline: Option<SearchDeadline>,
        ply: usize,
    ) -> Result<i32, SearchAborted> {
        let mut alpha = window.alpha;
        let beta = window.beta;
        stats.nodes += 1;
        if search_timed_out(deadline, stats.nodes) {
            return Err(SearchAborted::Timeout);
        }
        if is_draw_by_rule(position, &self.repetition_history) {
            return Ok(0);
        }

        let in_check = position.in_check(position.side_to_move);
        if !in_check {
            let stand_pat = position.evaluate();
            if stand_pat >= beta {
                return Ok(beta);
            }
            alpha = alpha.max(stand_pat);
        }

        let mut moves = if in_check {
            generate_legal_moves(position)
        } else {
            generate_legal_tactical_moves(position)
        };
        if moves.num_moves == 0 {
            return Ok(if in_check { -CHECKMATE_SCORE } else { alpha });
        }

        self.order_moves(position, &mut moves, None, ply);

        for m in moves.iter() {
            let undo = position.make_move_unchecked(m);
            self.repetition_history.push(position.history_hash());
            let value = self.quiescence(
                position,
                SearchWindow {
                    alpha: -beta,
                    beta: -alpha,
                },
                stats,
                deadline,
                ply + 1,
            );
            self.repetition_history.pop();
            position.undo_move_unchecked(m, undo);
            let value = -value?;

            if value >= beta {
                return Ok(beta);
            }
            if value > alpha {
                alpha = value;
            }
        }

        Ok(alpha)
    }
}

impl Default for Searcher {
    fn default() -> Self {
        Self::new()
    }
}

pub fn create_move(origin: Square, destination: Square) -> Move {
    create_flagged_move(origin, destination, MoveFlag::Quiet)
}

pub fn move_to_uci(m: Move) -> String {
    let mut uci = String::with_capacity(5);
    append_square_name(&mut uci, origin_square(m));
    append_square_name(&mut uci, destination_square(m));
    if let Some(kind) = promotion_kind(m) {
        uci.push(match kind {
            PieceKind::Knight => 'n',
            PieceKind::Bishop => 'b',
            PieceKind::Rook => 'r',
            PieceKind::Queen => 'q',
            PieceKind::Empty | PieceKind::Pawn | PieceKind::King => unreachable!(),
        });
    }
    uci
}

pub fn move_from_uci(position: &Position, uci: &str) -> Option<Move> {
    if !(uci.len() == 4 || uci.len() == 5) {
        return None;
    }

    let origin = parse_square_name(&uci[..2]).ok()?;
    let destination = parse_square_name(&uci[2..4]).ok()?;
    let promotion = match uci.as_bytes().get(4).copied() {
        Some(b'n') => Some(PieceKind::Knight),
        Some(b'b') => Some(PieceKind::Bishop),
        Some(b'r') => Some(PieceKind::Rook),
        Some(b'q') => Some(PieceKind::Queen),
        Some(_) => return None,
        None => None,
    };

    let mut scratch = *position;
    generate_legal_moves(&mut scratch).iter().find(|m| {
        origin_square(*m) == origin
            && destination_square(*m) == destination
            && promotion_kind(*m) == promotion
    })
}

pub fn is_selectable_piece(position: &Position, square: Square) -> bool {
    let piece = position.piece_at(square);
    piece != Piece::Empty && piece.color() == position.side_to_move()
}

pub fn legal_move_destinations(position: &Position, origin: Square) -> u64 {
    if !is_selectable_piece(position, origin) {
        return 0;
    }

    let mut scratch = *position;
    let legal_moves = generate_legal_moves(&mut scratch);
    let mut destinations = 0u64;
    for m in legal_moves.iter() {
        if origin_square(m) == origin {
            destinations |= to_bb(destination_square(m));
        }
    }

    destinations
}

pub fn player_move(position: &Position, origin: Square, destination: Square) -> Option<Move> {
    let mut scratch = *position;
    generate_legal_moves(&mut scratch).find(origin, destination)
}

pub fn try_player_move(position: &mut Position, origin: Square, destination: Square) -> bool {
    if let Some(m) = player_move(position, origin, destination) {
        position.do_move(m);
        true
    } else {
        false
    }
}

pub fn game_result(position: &Position, position_history: &[u64]) -> GameResult {
    if is_draw_by_rule(position, position_history) {
        return GameResult::Draw;
    }

    let mut scratch = *position;
    if generate_legal_moves(&mut scratch).num_moves != 0 {
        return GameResult::Ongoing;
    }

    if position.in_check(position.side_to_move) {
        match position.side_to_move {
            Color::White => GameResult::BlackWin,
            Color::Black => GameResult::WhiteWin,
        }
    } else {
        GameResult::Draw
    }
}

pub fn perft(position: &Position, depth: usize) -> u64 {
    let mut scratch = *position;
    perft_mut(&mut scratch, depth)
}

fn perft_mut(position: &mut Position, depth: usize) -> u64 {
    if depth == 0 {
        return 1;
    }

    let moves = generate_legal_moves(position);
    if depth == 1 {
        return moves.num_moves as u64;
    }

    let mut nodes = 0;
    for m in moves.iter() {
        let undo = position.make_move_unchecked(m);
        nodes += perft_mut(position, depth - 1);
        position.undo_move_unchecked(m, undo);
    }
    nodes
}

fn search_timed_out(deadline: Option<SearchDeadline>, nodes: u64) -> bool {
    deadline
        .is_some_and(|limit| nodes.is_multiple_of(TIME_CHECK_INTERVAL) && deadline_reached(limit))
}

#[cfg(not(target_arch = "wasm32"))]
fn deadline_after(duration: Duration) -> SearchDeadline {
    Instant::now() + duration
}

#[cfg(target_arch = "wasm32")]
fn deadline_after(duration: Duration) -> SearchDeadline {
    js_sys::Date::now() + duration.as_secs_f64() * 1000.0
}

#[cfg(not(target_arch = "wasm32"))]
fn deadline_reached(deadline: SearchDeadline) -> bool {
    Instant::now() >= deadline
}

#[cfg(target_arch = "wasm32")]
fn deadline_reached(deadline: SearchDeadline) -> bool {
    js_sys::Date::now() >= deadline
}

fn piece_from_code(code: i32) -> Option<Piece> {
    match code {
        0 => Some(Piece::Empty),
        1 => Some(Piece::WhitePawn),
        2 => Some(Piece::BlackPawn),
        3 => Some(Piece::WhiteKnight),
        4 => Some(Piece::BlackKnight),
        5 => Some(Piece::WhiteBishop),
        6 => Some(Piece::BlackBishop),
        7 => Some(Piece::WhiteRook),
        8 => Some(Piece::BlackRook),
        9 => Some(Piece::WhiteQueen),
        10 => Some(Piece::BlackQueen),
        11 => Some(Piece::WhiteKing),
        12 => Some(Piece::BlackKing),
        _ => None,
    }
}

fn color_from_code(code: u8) -> Option<Color> {
    match code {
        0 => Some(Color::White),
        1 => Some(Color::Black),
        _ => None,
    }
}

fn parse_square_name(square: &str) -> Result<Square, String> {
    let bytes = square.as_bytes();
    if bytes.len() != 2 {
        return Err(format!("invalid square: {square}"));
    }

    let file = match bytes[0] {
        b'a'..=b'h' => bytes[0] - b'a',
        _ => return Err(format!("invalid square file: {square}")),
    };
    let rank = match bytes[1] {
        b'1'..=b'8' => bytes[1] - b'1',
        _ => return Err(format!("invalid square rank: {square}")),
    };

    Ok((rank as u32) * 8 + file as u32)
}

fn append_square_name(buffer: &mut String, square: Square) {
    buffer.push((b'a' + (square % 8) as u8) as char);
    buffer.push((b'1' + (square / 8) as u8) as char);
}

pub fn search_best_move(
    board: &[i32],
    side_to_move: u8,
    castling_rights: u8,
    en_passant_square: i32,
    halfmove_clock: u32,
    position_history_hash_parts: &[u32],
    time_budget_ms: u32,
) -> i32 {
    if board.len() != 64 {
        return -1;
    }

    let Some(side_to_move) = color_from_code(side_to_move) else {
        return -1;
    };
    if castling_rights & !ALL_CASTLING_RIGHTS != 0 {
        return -1;
    }
    let en_passant_square = match en_passant_square {
        -1 => None,
        0..=63 => Some(en_passant_square as Square),
        _ => return -1,
    };
    let Ok(halfmove_clock) = u16::try_from(halfmove_clock) else {
        return -1;
    };
    if !position_history_hash_parts.len().is_multiple_of(2) {
        return -1;
    }

    let mut pieces = [Piece::Empty; 64];
    for (idx, &code) in board.iter().enumerate() {
        let Some(piece) = piece_from_code(code) else {
            return -1;
        };
        pieces[idx] = piece;
    }

    let position = Position::from_board_state(
        pieces,
        side_to_move,
        castling_rights,
        en_passant_square,
        halfmove_clock,
    );
    let mut position_history = Vec::with_capacity(position_history_hash_parts.len() / 2);
    for chunk in position_history_hash_parts.chunks_exact(2) {
        position_history.push(chunk[0] as u64 | ((chunk[1] as u64) << 32));
    }
    if !position_history.is_empty() && position_history.last() != Some(&position.history_hash()) {
        return -1;
    }

    let mut searcher = Searcher::new();
    searcher
        .best_move_with_time_budget(
            &position,
            &position_history,
            Duration::from_millis(time_budget_ms as u64),
        )
        .map(|(best_move, _)| best_move as i32)
        .unwrap_or(-1)
}

fn get_attacks(from: Square, position: &Position, piece_kind: PieceKind) -> Bitboard {
    match piece_kind {
        PieceKind::Knight => knight_attacks(from),
        PieceKind::Bishop => sliding_attacks(from, position.occupied(), &BISHOP_DIRECTIONS),
        PieceKind::Rook => sliding_attacks(from, position.occupied(), &ROOK_DIRECTIONS),
        PieceKind::Queen => {
            sliding_attacks(from, position.occupied(), &BISHOP_DIRECTIONS)
                | sliding_attacks(from, position.occupied(), &ROOK_DIRECTIONS)
        }
        PieceKind::King => king_attacks(from),
        PieceKind::Empty | PieceKind::Pawn => 0,
    }
}

fn generate_piece_moves(position: &Position, moves: &mut Moves, piece_kind: PieceKind) {
    let us = position.side_to_move;
    let own_pieces = position.colors[us as usize];
    let enemy_king = position.pieces(us.opposite(), PieceKind::King);
    let mut pieces = position.pieces(us, piece_kind);

    while pieces != 0 {
        let from = pop_square(&mut pieces);
        let mut targets = get_attacks(from, position, piece_kind) & !own_pieces & !enemy_king;
        while targets != 0 {
            moves.push(create_move(from, pop_square(&mut targets)));
        }
    }
}

fn push_promotion_moves(moves: &mut Moves, origin: Square, destination: Square) {
    moves.push(create_flagged_move(
        origin,
        destination,
        MoveFlag::PromoteQueen,
    ));
    moves.push(create_flagged_move(
        origin,
        destination,
        MoveFlag::PromoteKnight,
    ));
    moves.push(create_flagged_move(
        origin,
        destination,
        MoveFlag::PromoteRook,
    ));
    moves.push(create_flagged_move(
        origin,
        destination,
        MoveFlag::PromoteBishop,
    ));
}

fn generate_pawn_moves(position: &Position, moves: &mut Moves) {
    let us = position.side_to_move;
    let them = us.opposite();
    let up = pawn_push(us);
    let [up_left, up_right] = pawn_capture_directions(us);
    let empty = position.positions[PieceKind::Empty as usize];
    let pawns = position.pieces(us, PieceKind::Pawn);
    let enemy_king = position.pieces(them, PieceKind::King);
    let enemy_pieces = position.colors[them as usize] & !enemy_king;
    let direction_offset = |dir: Direction| dir as i32;
    let promotion_rank = match us {
        Color::White => 7,
        Color::Black => 0,
    };

    let mut single_pushes = shift(pawns, up) & empty;
    while single_pushes != 0 {
        let destination = pop_square(&mut single_pushes);
        let origin = (destination as i32 - direction_offset(up)) as Square;
        if destination / 8 == promotion_rank {
            push_promotion_moves(moves, origin, destination);
        } else {
            moves.push(create_move(origin, destination));
        }
    }

    let mut double_pushes = shift(
        (shift(pawns, up) & empty) & DOUBLE_PUSH_MASK[us as usize],
        up,
    ) & empty;
    while double_pushes != 0 {
        let destination = pop_square(&mut double_pushes);
        let origin = (destination as i32 - 2 * direction_offset(up)) as Square;
        moves.push(create_flagged_move(
            origin,
            destination,
            MoveFlag::DoublePawnPush,
        ));
    }

    let mut left_captures = shift(pawns, up_left) & enemy_pieces;
    while left_captures != 0 {
        let destination = pop_square(&mut left_captures);
        let origin = (destination as i32 - direction_offset(up_left)) as Square;
        if destination / 8 == promotion_rank {
            push_promotion_moves(moves, origin, destination);
        } else {
            moves.push(create_move(origin, destination));
        }
    }

    let mut right_captures = shift(pawns, up_right) & enemy_pieces;
    while right_captures != 0 {
        let destination = pop_square(&mut right_captures);
        let origin = (destination as i32 - direction_offset(up_right)) as Square;
        if destination / 8 == promotion_rank {
            push_promotion_moves(moves, origin, destination);
        } else {
            moves.push(create_move(origin, destination));
        }
    }

    if let Some(en_passant_square) = position.en_passant_square {
        let target = to_bb(en_passant_square);

        let mut left_en_passant = shift(pawns, up_left) & target;
        while left_en_passant != 0 {
            let destination = pop_square(&mut left_en_passant);
            let origin = (destination as i32 - direction_offset(up_left)) as Square;
            moves.push(create_flagged_move(
                origin,
                destination,
                MoveFlag::EnPassant,
            ));
        }

        let mut right_en_passant = shift(pawns, up_right) & target;
        while right_en_passant != 0 {
            let destination = pop_square(&mut right_en_passant);
            let origin = (destination as i32 - direction_offset(up_right)) as Square;
            moves.push(create_flagged_move(
                origin,
                destination,
                MoveFlag::EnPassant,
            ));
        }
    }
}

fn generate_king_moves(position: &Position, moves: &mut Moves) {
    let us = position.side_to_move;
    let own_pieces = position.colors[us as usize];
    let enemy_king = position.pieces(us.opposite(), PieceKind::King);
    let Some(from) = position.king_square(us) else {
        return;
    };

    let mut targets = king_attacks(from) & !own_pieces & !enemy_king;
    while targets != 0 {
        moves.push(create_move(from, pop_square(&mut targets)));
    }

    if position.in_check(us) {
        return;
    }

    let them = us.opposite();
    match us {
        Color::White => {
            if position.castling_rights & kingside_castle_right(us) != 0
                && position.piece_at(WHITE_KING_START) == Piece::WhiteKing
                && position.piece_at(WHITE_KINGSIDE_ROOK_START) == Piece::WhiteRook
                && position.piece_at(5) == Piece::Empty
                && position.piece_at(6) == Piece::Empty
                && !is_square_attacked(position, 5, them)
                && !is_square_attacked(position, 6, them)
            {
                moves.push(create_flagged_move(
                    WHITE_KING_START,
                    6,
                    MoveFlag::KingCastle,
                ));
            }

            if position.castling_rights & queenside_castle_right(us) != 0
                && position.piece_at(WHITE_KING_START) == Piece::WhiteKing
                && position.piece_at(WHITE_QUEENSIDE_ROOK_START) == Piece::WhiteRook
                && position.piece_at(1) == Piece::Empty
                && position.piece_at(2) == Piece::Empty
                && position.piece_at(3) == Piece::Empty
                && !is_square_attacked(position, 3, them)
                && !is_square_attacked(position, 2, them)
            {
                moves.push(create_flagged_move(
                    WHITE_KING_START,
                    2,
                    MoveFlag::QueenCastle,
                ));
            }
        }
        Color::Black => {
            if position.castling_rights & kingside_castle_right(us) != 0
                && position.piece_at(BLACK_KING_START) == Piece::BlackKing
                && position.piece_at(BLACK_KINGSIDE_ROOK_START) == Piece::BlackRook
                && position.piece_at(61) == Piece::Empty
                && position.piece_at(62) == Piece::Empty
                && !is_square_attacked(position, 61, them)
                && !is_square_attacked(position, 62, them)
            {
                moves.push(create_flagged_move(
                    BLACK_KING_START,
                    62,
                    MoveFlag::KingCastle,
                ));
            }

            if position.castling_rights & queenside_castle_right(us) != 0
                && position.piece_at(BLACK_KING_START) == Piece::BlackKing
                && position.piece_at(BLACK_QUEENSIDE_ROOK_START) == Piece::BlackRook
                && position.piece_at(57) == Piece::Empty
                && position.piece_at(58) == Piece::Empty
                && position.piece_at(59) == Piece::Empty
                && !is_square_attacked(position, 59, them)
                && !is_square_attacked(position, 58, them)
            {
                moves.push(create_flagged_move(
                    BLACK_KING_START,
                    58,
                    MoveFlag::QueenCastle,
                ));
            }
        }
    }
}

fn generate_pseudo_legal_moves(position: &Position) -> Moves {
    let mut moves = Moves::new();
    generate_piece_moves(position, &mut moves, PieceKind::Knight);
    generate_piece_moves(position, &mut moves, PieceKind::Bishop);
    generate_piece_moves(position, &mut moves, PieceKind::Rook);
    generate_piece_moves(position, &mut moves, PieceKind::Queen);
    generate_king_moves(position, &mut moves);
    generate_pawn_moves(position, &mut moves);
    moves
}

fn slider_attacks_square(
    position: &Position,
    square: Square,
    pieces: Bitboard,
    directions: &[Direction],
) -> bool {
    let target = to_bb(square);
    let occupied = position.occupied();
    let mut pieces = pieces;

    while pieces != 0 {
        let from = pop_square(&mut pieces);
        if sliding_attacks(from, occupied, directions) & target != 0 {
            return true;
        }
    }

    false
}

fn is_square_attacked(position: &Position, square: Square, attacker: Color) -> bool {
    let target = to_bb(square);
    let pawns = position.pieces(attacker, PieceKind::Pawn);
    let [up_left, up_right] = pawn_capture_directions(attacker);
    if (shift(pawns, up_left) | shift(pawns, up_right)) & target != 0 {
        return true;
    }

    let mut knights = position.pieces(attacker, PieceKind::Knight);
    while knights != 0 {
        let from = pop_square(&mut knights);
        if knight_attacks(from) & target != 0 {
            return true;
        }
    }

    if let Some(king_square) = position.king_square(attacker) {
        if king_attacks(king_square) & target != 0 {
            return true;
        }
    }

    slider_attacks_square(
        position,
        square,
        position.pieces(attacker, PieceKind::Bishop) | position.pieces(attacker, PieceKind::Queen),
        &BISHOP_DIRECTIONS,
    ) || slider_attacks_square(
        position,
        square,
        position.pieces(attacker, PieceKind::Rook) | position.pieces(attacker, PieceKind::Queen),
        &ROOK_DIRECTIONS,
    )
}

fn generate_legal_moves(position: &mut Position) -> Moves {
    let pseudo_legal_moves = generate_pseudo_legal_moves(position);
    let mut legal_moves = Moves::new();
    let side_to_move = position.side_to_move;

    for m in pseudo_legal_moves.iter() {
        let undo = position.make_move_unchecked(m);
        let is_legal = !position.in_check(side_to_move);
        position.undo_move_unchecked(m, undo);

        if is_legal {
            legal_moves.push(m);
        }
    }

    legal_moves
}

fn is_tactical_move(position: &Position, m: Move) -> bool {
    promotion_kind(m).is_some()
        || move_flag(m) == MoveFlag::EnPassant
        || position.by_piece[destination_square(m) as usize] != Piece::Empty
}

fn generate_legal_tactical_moves(position: &mut Position) -> Moves {
    let pseudo_legal_moves = generate_pseudo_legal_moves(position);
    let mut legal_moves = Moves::new();
    let side_to_move = position.side_to_move;

    for m in pseudo_legal_moves.iter() {
        if !is_tactical_move(position, m) {
            continue;
        }

        let undo = position.make_move_unchecked(m);
        let is_legal = !position.in_check(side_to_move);
        position.undo_move_unchecked(m, undo);

        if is_legal {
            legal_moves.push(m);
        }
    }

    legal_moves
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board_with(pieces: &[(Square, Piece)], side_to_move: Color) -> Position {
        let mut board = [Piece::Empty; 64];
        for &(square, piece) in pieces {
            board[square as usize] = piece;
        }
        Position::from_board_state(board, side_to_move, 0, None, 0)
    }

    fn position_from_fen(fen: &str) -> Position {
        Position::from_fen(fen).unwrap()
    }

    #[test]
    fn initial_position_has_twenty_legal_moves() {
        let mut position = Position::new();
        assert_eq!(generate_legal_moves(&mut position).num_moves, 20);
    }

    #[test]
    fn knight_attacks_from_the_center_cover_all_eight_squares() {
        let position = Position::new();
        let attacks = get_attacks(27, &position, PieceKind::Knight);
        let expected = [10, 12, 17, 21, 33, 37, 42, 44];

        assert_eq!(attacks.count_ones(), expected.len() as u32);
        for square in expected {
            assert_ne!(attacks & to_bb(square), 0);
        }
    }

    #[test]
    fn blocked_rook_move_is_not_legal() {
        let mut position = Position::new();
        assert!(generate_legal_moves(&mut position).find(0, 16).is_none());
    }

    #[test]
    fn evaluation_is_relative_to_the_side_to_move() {
        let white_to_move = board_with(
            &[
                (4, Piece::WhiteKing),
                (60, Piece::BlackKing),
                (3, Piece::WhiteQueen),
            ],
            Color::White,
        );
        let black_to_move = board_with(
            &[
                (4, Piece::WhiteKing),
                (60, Piece::BlackKing),
                (3, Piece::WhiteQueen),
            ],
            Color::Black,
        );

        assert!(white_to_move.evaluate() > 0);
        assert_eq!(white_to_move.evaluate(), -black_to_move.evaluate());
    }

    #[test]
    fn developed_knight_scores_higher_than_home_knight() {
        let home_knight = board_with(
            &[
                (4, Piece::WhiteKing),
                (60, Piece::BlackKing),
                (6, Piece::WhiteKnight),
            ],
            Color::White,
        );
        let developed_knight = board_with(
            &[
                (4, Piece::WhiteKing),
                (60, Piece::BlackKing),
                (21, Piece::WhiteKnight),
            ],
            Color::White,
        );

        assert!(developed_knight.evaluate() > home_knight.evaluate());
    }

    #[test]
    fn castled_king_scores_higher_than_uncastled_king() {
        let uncastled = position_from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1");
        let castled = position_from_fen("4k3/8/8/8/8/8/8/R4RK1 w - - 0 1");

        assert!(castled.evaluate() > uncastled.evaluate());
    }

    #[test]
    fn active_rook_scores_higher_than_corner_rook() {
        let corner_rook = board_with(
            &[
                (7, Piece::WhiteKing),
                (63, Piece::BlackKing),
                (0, Piece::WhiteRook),
            ],
            Color::White,
        );
        let active_rook = board_with(
            &[
                (7, Piece::WhiteKing),
                (63, Piece::BlackKing),
                (27, Piece::WhiteRook),
            ],
            Color::White,
        );

        assert!(active_rook.evaluate() > corner_rook.evaluate());
    }

    #[test]
    fn engine_prefers_the_winning_capture() {
        let mut searcher = Searcher::new();
        let mut position = board_with(
            &[
                (7, Piece::WhiteKing),
                (0, Piece::WhiteRook),
                (56, Piece::BlackQueen),
                (63, Piece::BlackKing),
            ],
            Color::White,
        );
        let position_history = [position.history_hash()];

        assert!(position.in_check(Color::White));
        assert!(searcher.make_move(&mut position, &position_history));
        assert_eq!(position.by_piece[56], Piece::WhiteRook);
        assert_eq!(position.by_piece[0], Piece::Empty);
    }

    #[test]
    fn make_unmake_restores_hash_and_board() {
        let mut position = Position::new();
        let original = position;
        let m = create_move(12, 28);

        let undo = position.make_move_unchecked(m);
        position.undo_move_unchecked(m, undo);

        assert_eq!(position.hash, original.hash);
        for square in 0..64 {
            assert_eq!(position.piece_at(square), original.piece_at(square));
        }
        assert_eq!(position.side_to_move(), original.side_to_move());
        assert_eq!(
            position.castling_rights_bits(),
            original.castling_rights_bits()
        );
        assert_eq!(position.en_passant_square(), original.en_passant_square());
        assert_eq!(position.halfmove_clock(), original.halfmove_clock());
    }

    #[test]
    fn castling_moves_are_generated_when_legal() {
        let mut position = position_from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        let legal_moves = generate_legal_moves(&mut position);

        assert!(legal_moves.find(4, 6).is_some());
        assert!(legal_moves.find(4, 2).is_some());
    }

    #[test]
    fn en_passant_capture_is_legal() {
        let mut position = position_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
        let en_passant = generate_legal_moves(&mut position)
            .find(36, 43)
            .expect("en passant should be legal");

        position.do_move(en_passant);

        assert_eq!(position.piece_at(43), Piece::WhitePawn);
        assert_eq!(position.piece_at(35), Piece::Empty);
        assert_eq!(position.piece_at(36), Piece::Empty);
    }

    #[test]
    fn promotions_generate_all_four_choices() {
        let mut position = position_from_fen("7k/P7/8/8/8/8/8/4K3 w - - 0 1");
        let promotion_count = generate_legal_moves(&mut position)
            .iter()
            .filter(|m| origin_square(*m) == 48 && destination_square(*m) == 56)
            .count();

        assert_eq!(promotion_count, 4);
    }

    #[test]
    fn halfmove_clock_increments_and_resets() {
        let mut quiet_position = position_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 17 1");
        quiet_position.do_move(create_move(4, 12));
        assert_eq!(quiet_position.halfmove_clock(), 18);

        let mut pawn_position = position_from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 17 1");
        pawn_position.do_move(create_move(12, 20));
        assert_eq!(pawn_position.halfmove_clock(), 0);

        let mut capture_position = position_from_fen("4k3/8/8/8/8/8/4r3/4K3 w - - 17 1");
        capture_position.do_move(create_move(4, 12));
        assert_eq!(capture_position.halfmove_clock(), 0);
    }

    #[test]
    fn fifty_move_draw_is_detected() {
        let position = position_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 100 1");
        let position_history = [position.history_hash()];

        assert!(is_draw_by_rule(&position, &position_history));
    }

    #[test]
    fn threefold_repetition_is_detected() {
        let mut position = position_from_fen("4k1n1/8/8/8/8/8/8/4K1N1 w - - 0 1");
        let mut position_history = vec![position.history_hash()];

        for m in [
            create_move(6, 21),
            create_move(62, 45),
            create_move(21, 6),
            create_move(45, 62),
            create_move(6, 21),
            create_move(62, 45),
            create_move(21, 6),
            create_move(45, 62),
        ] {
            position.do_move(m);
            position_history.push(position.history_hash());
        }

        assert!(is_draw_by_rule(&position, &position_history));
    }

    #[test]
    fn engine_does_not_search_rule_draws() {
        let mut searcher = Searcher::new();
        let position = position_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 100 1");
        let position_history = [position.history_hash()];

        assert!(searcher
            .best_move_with_time_budget(&position, &position_history, Duration::from_millis(1),)
            .is_none());
    }

    #[test]
    fn quiescence_avoids_a_poisoned_rook_capture() {
        let mut searcher = Searcher::new();
        let mut position = position_from_fen("6k1/8/3b4/4r3/8/8/4Q3/6K1 w - - 0 1");

        let (best_move, _) = searcher
            .search_root(&mut position, 1, None)
            .unwrap()
            .unwrap();

        assert_ne!(move_to_uci(best_move), "e2e5");
    }

    #[test]
    fn uci_move_round_trips_a_quiet_move() {
        let position = Position::new();
        let m = move_from_uci(&position, "e2e4").expect("uci move should parse");

        assert_eq!(move_to_uci(m), "e2e4");
    }

    #[test]
    fn uci_move_round_trips_a_promotion() {
        let position = position_from_fen("7k/P7/8/8/8/8/8/4K3 w - - 0 1");
        let m = move_from_uci(&position, "a7a8q").expect("promotion should parse");

        assert_eq!(move_to_uci(m), "a7a8q");
    }
}
