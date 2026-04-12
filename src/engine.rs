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

pub const SEARCH_DEPTH: usize = 3;
const CHECKMATE_SCORE: i32 = 100_000;
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
}

#[derive(Debug, Copy, Clone, Default)]
pub struct SearchStats {
    pub nodes: u64,
    pub score: i32,
}

fn set_square(bitboard: &mut Bitboard, square: Square) {
    *bitboard |= 1u64 << square;
}

fn unset_square(bitboard: &mut Bitboard, square: Square) {
    *bitboard &= !(1u64 << square);
}

fn destination_square(m: Move) -> Square {
    m & 0x3F
}

fn origin_square(m: Move) -> Square {
    (m >> 6) & 0x3F
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
        };

        for (i, piece) in position.by_piece.iter().copied().enumerate() {
            let square = i as Square;
            let pos = &mut position.positions[piece.kind() as usize];
            set_square(pos, square);
            if piece != Piece::Empty {
                set_square(&mut position.colors[piece.color() as usize], square);
            }
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

        let us = self.side_to_move as usize;
        let them = self.side_to_move.opposite() as usize;

        unset_square(&mut self.colors[us], origin);
        set_square(&mut self.colors[us], destination);
        if target_piece != Piece::Empty {
            unset_square(&mut self.colors[them], destination);
        }

        set_square(&mut self.positions[PieceKind::Empty as usize], origin);
        let moving_piece_kind = moving_piece.kind() as usize;
        let target_piece_kind = target_piece.kind() as usize;

        unset_square(&mut self.positions[moving_piece_kind], origin);
        unset_square(&mut self.positions[target_piece_kind], destination);

        let destination_rank = destination / 8;
        let moved_piece = if moving_piece.kind() == PieceKind::Pawn
            && (destination_rank == 0 || destination_rank == 7)
        {
            Piece::from_kind_color(PieceKind::Queen, self.side_to_move)
        } else {
            moving_piece
        };

        set_square(
            &mut self.positions[moved_piece.kind() as usize],
            destination,
        );

        self.by_piece[destination as usize] = moved_piece;
        self.by_piece[origin as usize] = Piece::Empty;
        self.side_to_move = self.side_to_move.opposite();

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
        for i in 0..64 {
            let piece = self.by_piece[i as usize];
            let bb: u64 = 1u64 << i as u64;
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

#[derive(Copy, Clone)]
struct Moves {
    moves: [Move; 256],
    num_moves: usize,
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

struct MoveIter<'a> {
    moves: &'a Moves,
    idx: usize,
}

impl<'a> Iterator for MoveIter<'a> {
    type Item = Move;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx == self.moves.num_moves {
            return None;
        }
        let i = self.idx;
        self.idx += 1;
        Some(self.moves.moves[i])
    }
}

pub fn create_move(origin: Square, destination: Square) -> Move {
    let mut m = destination;
    m |= origin << 6;
    m
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

fn generate_legal_moves(position: &Position) -> Moves {
    let pseudo_legal_moves = generate_pseudo_legal_moves(position);
    let mut legal_moves = Moves::new();
    let side_to_move = position.side_to_move;

    for m in pseudo_legal_moves.iter() {
        let mut next = *position;
        next.do_move(m);
        if !next.in_check(side_to_move) {
            legal_moves.push(m);
        }
    }

    legal_moves
}

fn negamax(position: &Position, depth: usize, mut alpha: i32, beta: i32) -> i32 {
    if depth == 0 {
        return position.evaluate();
    }

    let moves = generate_legal_moves(position);
    if moves.num_moves == 0 {
        return if position.in_check(position.side_to_move) {
            -CHECKMATE_SCORE + depth as i32
        } else {
            0
        };
    }

    let mut best_value = -CHECKMATE_SCORE;
    for m in moves.iter() {
        let mut next = *position;
        next.do_move(m);
        let value = -negamax(&next, depth - 1, -beta, -alpha);
        if value > best_value {
            best_value = value;
        }
        if value > alpha {
            alpha = value;
        }
        if alpha >= beta {
            break;
        }
    }

    best_value
}

fn negamax_with_stats(
    position: &Position,
    depth: usize,
    mut alpha: i32,
    beta: i32,
    stats: &mut SearchStats,
) -> i32 {
    stats.nodes += 1;

    if depth == 0 {
        return position.evaluate();
    }

    let moves = generate_legal_moves(position);
    if moves.num_moves == 0 {
        return if position.in_check(position.side_to_move) {
            -CHECKMATE_SCORE + depth as i32
        } else {
            0
        };
    }

    let mut best_value = -CHECKMATE_SCORE;
    for m in moves.iter() {
        let mut next = *position;
        next.do_move(m);
        let value = -negamax_with_stats(&next, depth - 1, -beta, -alpha, stats);
        if value > best_value {
            best_value = value;
        }
        if value > alpha {
            alpha = value;
        }
        if alpha >= beta {
            break;
        }
    }

    best_value
}

pub fn make_move(position: &mut Position) -> bool {
    let moves = generate_legal_moves(position);
    if moves.num_moves == 0 {
        return false;
    }

    let mut best_move = moves.moves[0];
    let mut best_value = -CHECKMATE_SCORE;
    for m in moves.iter() {
        let mut next = *position;
        next.do_move(m);
        let value = -negamax(
            &next,
            SEARCH_DEPTH.saturating_sub(1),
            -CHECKMATE_SCORE,
            CHECKMATE_SCORE,
        );
        if value > best_value {
            best_move = m;
            best_value = value;
        }
    }

    position.do_move(best_move);
    true
}

pub fn try_player_move(position: &mut Position, origin: Square, destination: Square) -> bool {
    let legal_moves = generate_legal_moves(position);
    if let Some(m) = legal_moves.find(origin, destination) {
        position.do_move(m);
        return true;
    }
    false
}

pub fn search_root_with_stats(position: &Position, depth: usize) -> Option<SearchStats> {
    let moves = generate_legal_moves(position);
    if moves.num_moves == 0 {
        return None;
    }

    let mut stats = SearchStats::default();
    let mut best_score = -CHECKMATE_SCORE;

    for m in moves.iter() {
        let mut next = *position;
        next.do_move(m);
        let value = -negamax_with_stats(
            &next,
            depth.saturating_sub(1),
            -CHECKMATE_SCORE,
            CHECKMATE_SCORE,
            &mut stats,
        );
        if value > best_score {
            best_score = value;
        }
    }

    stats.score = best_score;
    Some(stats)
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
        let position = Position::new();
        assert_eq!(generate_legal_moves(&position).num_moves, 20);
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
        let position = Position::new();
        assert!(generate_legal_moves(&position).find(0, 16).is_none());
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
        assert!(make_move(&mut position));
        assert_eq!(position.by_piece[56], Piece::WhiteRook);
        assert_eq!(position.by_piece[0], Piece::Empty);
    }
}
