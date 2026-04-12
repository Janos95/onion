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

pub const SEARCH_TIME_BUDGET_MS: u32 = 2_000;
const CHECKMATE_SCORE: i32 = 100_000;
const MAX_SEARCH_DEPTH: usize = 64;
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
const ZOBRIST_SEED_INC: u64 = 0x9E37_79B9_7F4A_7C15;

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
    hash: u64,
}

#[derive(Debug, Copy, Clone, Default)]
pub struct SearchStats {
    pub nodes: u64,
    pub score: i32,
}

#[derive(Copy, Clone)]
struct Undo {
    moving_piece: Piece,
    placed_piece: Piece,
    captured_piece: Piece,
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
}

#[derive(Copy, Clone)]
enum SearchAborted {
    Timeout,
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

const ZOBRIST_PIECES: [[u64; 64]; 13] = generate_piece_hashes();
const ZOBRIST_SIDE_TO_MOVE: u64 = mix64(0xCAFEBABE_DEADBEEF);

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

fn destination_square(m: Move) -> Square {
    m & 0x3F
}

fn origin_square(m: Move) -> Square {
    (m >> 6) & 0x3F
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
        Position::from_board(INITIAL_PIECES, Color::White)
    }

    fn from_board(by_piece: [Piece; 64], side_to_move: Color) -> Position {
        let mut position = Position {
            positions: [0; 7],
            colors: [0; 2],
            by_piece,
            side_to_move,
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

        position
    }

    pub fn piece_at(&self, square: Square) -> Piece {
        self.by_piece[square as usize]
    }

    pub fn side_to_move(&self) -> Color {
        self.side_to_move
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
        let moving_piece = self.by_piece[origin as usize];
        let captured_piece = self.by_piece[destination as usize];
        let destination_rank = destination / 8;
        let placed_piece = if moving_piece.kind() == PieceKind::Pawn
            && (destination_rank == 0 || destination_rank == 7)
        {
            Piece::from_kind_color(PieceKind::Queen, self.side_to_move)
        } else {
            moving_piece
        };

        self.remove_piece(origin, moving_piece);
        if captured_piece != Piece::Empty {
            self.remove_piece(destination, captured_piece);
        }
        self.add_piece(destination, placed_piece);

        self.side_to_move = self.side_to_move.opposite();
        self.hash ^= ZOBRIST_SIDE_TO_MOVE;

        Undo {
            moving_piece,
            placed_piece,
            captured_piece,
        }
    }

    fn undo_move_unchecked(&mut self, m: Move, undo: Undo) {
        let origin = origin_square(m);
        let destination = destination_square(m);

        self.side_to_move = self.side_to_move.opposite();
        self.hash ^= ZOBRIST_SIDE_TO_MOVE;

        self.remove_piece(destination, undo.placed_piece);
        self.add_piece(origin, undo.moving_piece);
        if undo.captured_piece != Piece::Empty {
            self.add_piece(destination, undo.captured_piece);
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

        let mut value = 0;
        value += (self.positions[PieceKind::Pawn as usize] & color_mask).count_ones() as i32 * 100;
        value +=
            (self.positions[PieceKind::Knight as usize] & color_mask).count_ones() as i32 * 320;
        value +=
            (self.positions[PieceKind::Bishop as usize] & color_mask).count_ones() as i32 * 330;
        value += (self.positions[PieceKind::Rook as usize] & color_mask).count_ones() as i32 * 500;
        value += (self.positions[PieceKind::Queen as usize] & color_mask).count_ones() as i32 * 900;
        value +=
            (self.positions[PieceKind::King as usize] & color_mask).count_ones() as i32 * 20000;
        value
    }

    fn evaluate(&self) -> i32 {
        self.material(self.side_to_move) - self.material(self.side_to_move.opposite())
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
        }
    }

    pub fn make_move(&mut self, position: &mut Position) -> bool {
        let Some((best_move, _)) = self.best_move_with_time_budget(
            position,
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
        depth: usize,
    ) -> Option<SearchStats> {
        let mut scratch = *position;
        self.search_root(&mut scratch, depth, None)
            .ok()
            .flatten()
            .map(|(_, stats)| stats)
    }

    pub fn best_move_with_time_budget(
        &mut self,
        position: &Position,
        time_budget: Duration,
    ) -> Option<(Move, SearchStats)> {
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

        let tt_move = self.tt.probe(position.hash).map(|entry| entry.best_move);
        order_moves(position, &mut moves, tt_move);

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
            let value = -self.negamax(
                position,
                depth.saturating_sub(1),
                -beta,
                -alpha,
                &mut stats,
                deadline,
            )?;
            position.undo_move_unchecked(m, undo);

            if value > best_score {
                best_score = value;
                best_move = m;
            }

            if value > alpha {
                alpha = value;
            }
        }

        self.tt
            .store(position.hash, depth, best_score, Bound::Exact, best_move);

        stats.score = best_score;
        Ok(Some((best_move, stats)))
    }

    fn negamax(
        &mut self,
        position: &mut Position,
        depth: usize,
        mut alpha: i32,
        beta: i32,
        stats: &mut SearchStats,
        deadline: Option<SearchDeadline>,
    ) -> Result<i32, SearchAborted> {
        stats.nodes += 1;
        if search_timed_out(deadline, stats.nodes) {
            return Err(SearchAborted::Timeout);
        }

        let original_alpha = alpha;
        if depth == 0 {
            return Ok(position.evaluate());
        }

        let tt_entry = self.tt.probe(position.hash);
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

        order_moves(position, &mut moves, tt_entry.map(|entry| entry.best_move));

        let mut best_value = -CHECKMATE_SCORE;
        let mut best_move = 0;

        for m in moves.iter() {
            let undo = position.make_move_unchecked(m);
            let value = -self.negamax(position, depth - 1, -beta, -alpha, stats, deadline)?;
            position.undo_move_unchecked(m, undo);

            if value > best_value {
                best_value = value;
                best_move = m;
            }
            if value > alpha {
                alpha = value;
            }
            if alpha >= beta {
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
            .store(position.hash, depth, best_value, bound, best_move);
        Ok(best_value)
    }
}

impl Default for Searcher {
    fn default() -> Self {
        Self::new()
    }
}

pub fn create_move(origin: Square, destination: Square) -> Move {
    let mut m = destination;
    m |= origin << 6;
    m
}

pub fn make_move(position: &mut Position) -> bool {
    Searcher::new().make_move(position)
}

pub fn try_player_move(position: &mut Position, origin: Square, destination: Square) -> bool {
    let legal_moves = generate_legal_moves(position);
    if let Some(m) = legal_moves.find(origin, destination) {
        position.do_move(m);
        true
    } else {
        false
    }
}

pub fn search_root_with_stats(position: &Position, depth: usize) -> Option<SearchStats> {
    Searcher::new().search_root_with_stats(position, depth)
}

fn search_timed_out(deadline: Option<SearchDeadline>, nodes: u64) -> bool {
    deadline.is_some_and(|limit| nodes % TIME_CHECK_INTERVAL == 0 && deadline_reached(limit))
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

pub fn search_best_move_for_board(board: &[i32], side_to_move: u8, time_budget_ms: u32) -> i32 {
    if board.len() != 64 {
        return -1;
    }

    let Some(side_to_move) = color_from_code(side_to_move) else {
        return -1;
    };

    let mut pieces = [Piece::Empty; 64];
    for (idx, &code) in board.iter().enumerate() {
        let Some(piece) = piece_from_code(code) else {
            return -1;
        };
        pieces[idx] = piece;
    }

    let position = Position::from_board(pieces, side_to_move);
    let mut searcher = Searcher::new();
    searcher
        .best_move_with_time_budget(&position, Duration::from_millis(time_budget_ms as u64))
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

    let mut single_pushes = shift(pawns, up) & empty;
    while single_pushes != 0 {
        let destination = pop_square(&mut single_pushes);
        let origin = (destination as i32 - direction_offset(up)) as Square;
        moves.push(create_move(origin, destination));
    }

    let mut double_pushes = shift(
        (shift(pawns, up) & empty) & DOUBLE_PUSH_MASK[us as usize],
        up,
    ) & empty;
    while double_pushes != 0 {
        let destination = pop_square(&mut double_pushes);
        let origin = (destination as i32 - 2 * direction_offset(up)) as Square;
        moves.push(create_move(origin, destination));
    }

    let mut left_captures = shift(pawns, up_left) & enemy_pieces;
    while left_captures != 0 {
        let destination = pop_square(&mut left_captures);
        let origin = (destination as i32 - direction_offset(up_left)) as Square;
        moves.push(create_move(origin, destination));
    }

    let mut right_captures = shift(pawns, up_right) & enemy_pieces;
    while right_captures != 0 {
        let destination = pop_square(&mut right_captures);
        let origin = (destination as i32 - direction_offset(up_right)) as Square;
        moves.push(create_move(origin, destination));
    }
}

fn generate_pseudo_legal_moves(position: &Position) -> Moves {
    let mut moves = Moves::new();
    generate_piece_moves(position, &mut moves, PieceKind::Knight);
    generate_piece_moves(position, &mut moves, PieceKind::Bishop);
    generate_piece_moves(position, &mut moves, PieceKind::Rook);
    generate_piece_moves(position, &mut moves, PieceKind::Queen);
    generate_piece_moves(position, &mut moves, PieceKind::King);
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

fn move_score(position: &Position, m: Move, tt_move: Option<Move>) -> i32 {
    if tt_move.is_some_and(|candidate| candidate == m) {
        return i32::MAX;
    }

    let origin = origin_square(m);
    let destination = destination_square(m);
    let moving_piece = position.by_piece[origin as usize];
    let captured_piece = position.by_piece[destination as usize];

    let mut score = 0;
    if captured_piece != Piece::Empty {
        score += 10_000;
        score += piece_value(captured_piece.kind()) * 16 - piece_value(moving_piece.kind());
    }
    if moving_piece.kind() == PieceKind::Pawn && (destination / 8 == 0 || destination / 8 == 7) {
        score += 8_000;
    }
    score
}

fn order_moves(position: &Position, moves: &mut Moves, tt_move: Option<Move>) {
    let mut scores = [0; 256];
    for idx in 0..moves.num_moves {
        scores[idx] = move_score(position, moves.moves[idx], tt_move);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn board_with(pieces: &[(Square, Piece)], side_to_move: Color) -> Position {
        let mut board = [Piece::Empty; 64];
        for &(square, piece) in pieces {
            board[square as usize] = piece;
        }
        Position::from_board(board, side_to_move)
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

        assert_eq!(white_to_move.evaluate(), 900);
        assert_eq!(black_to_move.evaluate(), -900);
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

        assert!(position.in_check(Color::White));
        assert!(searcher.make_move(&mut position));
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
    }
}
