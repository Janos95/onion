use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{Event, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

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

const SEARCH_DEPTH: usize = 3;
const CHECKMATE_SCORE: i32 = 100_000;
const KNIGHT_DELTAS: [(i32, i32); 8] = [
    (2, 1),
    (2, -1),
    (1, 2),
    (1, -2),
    (-1, 2),
    (-1, -2),
    (-2, 1),
    (-2, -1),
];
const KING_DELTAS: [(i32, i32); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];
const BISHOP_DIRECTIONS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const ROOK_DIRECTIONS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

// chess colors
#[repr(u8)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Color {
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
#[derive(Copy, Clone, Eq, PartialEq)]
enum PieceKind {
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
enum Piece {
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

type Bitboard = u64;
type Square = u32;

/// bits 0-5: destination square, bits 6-11: origin square
type Move = u32;

fn destination_square(m: Move) -> Square {
    return m & 0x3F;
}

fn origin_square(m: Move) -> Square {
    return (m >> 6) & 0x3F;
}

#[derive(Copy, Clone)]
struct Position {
    positions: [Bitboard; 7], // empty, pawn, knight, bishop, rook, queen, king
    colors: [Bitboard; 2],    // white, black
    by_piece: [Piece; 64],
    side_to_move: Color,
}

fn set_square(bitboard: &mut Bitboard, square: Square) {
    *bitboard |= 1u64 << square;
}

fn unset_square(bitboard: &mut Bitboard, square: Square) {
    *bitboard &= !(1u64 << square);
}

impl Position {
    fn new() -> Position {
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

        //println!("white pieces {}", position.colors[0]);
        //println!("black pieces {}", position.colors[1]);

        position
    }

    fn occupied(&self) -> Bitboard {
        self.colors[Color::White as usize] | self.colors[Color::Black as usize]
    }

    fn pieces(&self, c: Color, kind: PieceKind) -> Bitboard {
        self.colors[c as usize] & self.positions[kind as usize]
    }

    fn king_square(&self, c: Color) -> Option<Square> {
        let kings = self.pieces(c, PieceKind::King);
        if kings == 0 {
            None
        } else {
            Some(kings.trailing_zeros())
        }
    }

    fn do_move(&mut self, m: Move) {
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

        // color bitboards
        unset_square(&mut self.colors[us], origin);
        set_square(&mut self.colors[us], destination);
        if target_piece != Piece::Empty {
            unset_square(&mut self.colors[them], destination);
        }

        // origin becomes empty, destination stops being empty if needed
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

        // explicit piece array
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
                let c = piece.color();
                if self.colors[c as usize] & bb == 0 {
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

fn pop_square(board: &mut Bitboard) -> Square {
    let s = board.trailing_zeros();
    *board = *board & (*board - 1);
    s
}

#[derive(Copy, Clone)]
struct Moves {
    moves: [Move; 256],
    num_moves: usize,
}

fn create_move(origin: Square, destination: Square) -> Move {
    let mut m = destination;
    m |= origin << 6;
    m
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

fn to_bb(s: Square) -> Bitboard {
    1u64 << s
}

fn file_of(square: Square) -> i32 {
    (square % 8) as i32
}

fn rank_of(square: Square) -> i32 {
    (square / 8) as i32
}

fn square_from_coords(file: i32, rank: i32) -> Option<Square> {
    if !(0..8).contains(&file) || !(0..8).contains(&rank) {
        return None;
    }
    Some((rank * 8 + file) as Square)
}

fn jump_attacks(from: Square, deltas: &[(i32, i32)]) -> Bitboard {
    let file = file_of(from);
    let rank = rank_of(from);
    let mut attacks = 0;
    for (df, dr) in deltas {
        if let Some(square) = square_from_coords(file + df, rank + dr) {
            attacks |= to_bb(square);
        }
    }
    attacks
}

fn sliding_attacks(from: Square, occupied: Bitboard, directions: &[(i32, i32)]) -> Bitboard {
    let mut attacks = 0;
    for (df, dr) in directions {
        let mut file = file_of(from) + df;
        let mut rank = rank_of(from) + dr;
        while let Some(square) = square_from_coords(file, rank) {
            attacks |= to_bb(square);
            if occupied & to_bb(square) != 0 {
                break;
            }
            file += df;
            rank += dr;
        }
    }
    attacks
}

fn get_attacks(from: Square, position: &Position, piece_kind: PieceKind) -> Bitboard {
    match piece_kind {
        PieceKind::Knight => jump_attacks(from, &KNIGHT_DELTAS),
        PieceKind::Bishop => sliding_attacks(from, position.occupied(), &BISHOP_DIRECTIONS),
        PieceKind::Rook => sliding_attacks(from, position.occupied(), &ROOK_DIRECTIONS),
        PieceKind::Queen => {
            sliding_attacks(from, position.occupied(), &BISHOP_DIRECTIONS)
                | sliding_attacks(from, position.occupied(), &ROOK_DIRECTIONS)
        }
        PieceKind::King => jump_attacks(from, &KING_DELTAS),
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
    let occupied = position.occupied();
    let enemy_king = position.pieces(them, PieceKind::King);
    let enemy_pieces = position.colors[them as usize] & !enemy_king;
    let forward = if us == Color::White { 1 } else { -1 };
    let start_rank = if us == Color::White { 1 } else { 6 };
    let mut pawns = position.pieces(us, PieceKind::Pawn);

    while pawns != 0 {
        let from = pop_square(&mut pawns);
        let file = file_of(from);
        let rank = rank_of(from);

        if let Some(destination) = square_from_coords(file, rank + forward) {
            if occupied & to_bb(destination) == 0 {
                moves.push(create_move(from, destination));

                if rank == start_rank {
                    if let Some(double_destination) = square_from_coords(file, rank + 2 * forward) {
                        if occupied & to_bb(double_destination) == 0 {
                            moves.push(create_move(from, double_destination));
                        }
                    }
                }
            }
        }

        for file_delta in [-1, 1] {
            if let Some(destination) = square_from_coords(file + file_delta, rank + forward) {
                if enemy_pieces & to_bb(destination) != 0 {
                    moves.push(create_move(from, destination));
                }
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
    generate_piece_moves(position, &mut moves, PieceKind::King);
    generate_pawn_moves(position, &mut moves);
    moves
}

fn slider_attacks_square(
    position: &Position,
    square: Square,
    attacker: Color,
    directions: &[(i32, i32)],
    primary: PieceKind,
    secondary: PieceKind,
) -> bool {
    for (df, dr) in directions {
        let mut file = file_of(square) + df;
        let mut rank = rank_of(square) + dr;

        while let Some(candidate) = square_from_coords(file, rank) {
            let piece = position.by_piece[candidate as usize];
            if piece != Piece::Empty {
                if piece.color() == attacker
                    && (piece.kind() == primary || piece.kind() == secondary)
                {
                    return true;
                }
                break;
            }

            file += df;
            rank += dr;
        }
    }
    false
}

fn is_square_attacked(position: &Position, square: Square, attacker: Color) -> bool {
    let target = to_bb(square);
    let pawn_rank = rank_of(square) + if attacker == Color::White { -1 } else { 1 };

    for file_delta in [-1, 1] {
        if let Some(candidate) = square_from_coords(file_of(square) + file_delta, pawn_rank) {
            let piece = position.by_piece[candidate as usize];
            if piece != Piece::Empty && piece.color() == attacker && piece.kind() == PieceKind::Pawn
            {
                return true;
            }
        }
    }

    let mut knights = position.pieces(attacker, PieceKind::Knight);
    while knights != 0 {
        let from = pop_square(&mut knights);
        if jump_attacks(from, &KNIGHT_DELTAS) & target != 0 {
            return true;
        }
    }

    if let Some(king_square) = position.king_square(attacker) {
        if jump_attacks(king_square, &KING_DELTAS) & target != 0 {
            return true;
        }
    }

    slider_attacks_square(
        position,
        square,
        attacker,
        &BISHOP_DIRECTIONS,
        PieceKind::Bishop,
        PieceKind::Queen,
    ) || slider_attacks_square(
        position,
        square,
        attacker,
        &ROOK_DIRECTIONS,
        PieceKind::Rook,
        PieceKind::Queen,
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
        let mut new_position = *position;
        new_position.do_move(m);
        let value = -negamax(&new_position, depth - 1, -beta, -alpha);
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

fn make_move(position: &mut Position) -> bool {
    let moves = generate_legal_moves(position);
    if moves.num_moves == 0 {
        return false;
    }

    let mut best_move = moves.moves[0];
    let mut best_value = -CHECKMATE_SCORE;
    for m in moves.iter() {
        let mut next = *position;
        next.do_move(m);
        let v = -negamax(
            &next,
            SEARCH_DEPTH.saturating_sub(1),
            -CHECKMATE_SCORE,
            CHECKMATE_SCORE,
        );
        if v > best_value {
            best_move = m;
            best_value = v;
        }
    }

    position.do_move(best_move);
    true
}

fn try_player_move(position: &mut Position, origin: Square, destination: Square) -> bool {
    let legal_moves = generate_legal_moves(position);
    if let Some(m) = legal_moves.find(origin, destination) {
        position.do_move(m);
        return true;
    }
    false
}

fn mouse_to_square(
    position: PhysicalPosition<f64>,
    window_size: PhysicalSize<u32>,
) -> Option<Square> {
    if window_size.width == 0 || window_size.height == 0 {
        return None;
    }

    let x = (position.x / window_size.width as f64 * 8.0).floor() as i32;
    let y = 7 - (position.y / window_size.height as f64 * 8.0).floor() as i32;

    if !(0..8).contains(&x) || !(0..8).contains(&y) {
        return None;
    }

    Some((x + y * 8) as Square)
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

#[repr(C, align(16))]
#[derive(Debug, Copy, Clone)]
struct GpuBoard {
    pieces: [i32; 64],
}

unsafe impl Zeroable for GpuBoard {}
unsafe impl Pod for GpuBoard {}

impl GpuBoard {
    fn new(pos: &Position) -> GpuBoard {
        let mut board = GpuBoard { pieces: [0; 64] };
        for i in 0..64 {
            board.pieces[i] = pos.by_piece[i] as i32;
        }
        board
    }

    fn as_byte_slice(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let size = window.inner_size();

    let instance = wgpu::Instance::default();

    let surface = unsafe { instance.create_surface(&window) }.unwrap();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            // Request an adapter which can render to our surface
            compatible_surface: Some(&surface),
        })
        .await
        .expect("Failed to find an appropriate adapter");

    // Create the logical device and command queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
            },
            None,
        )
        .await
        .expect("Failed to create device");

    // Load the shaders from disk
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    let mut position = Position::new();
    let gpu_board = GpuBoard::new(&position);

    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Uniform Buffer"),
        contents: gpu_board.as_byte_slice(),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<GpuBoard>() as u64),
            },
            count: None,
        }],
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buf.as_entire_binding(),
        }],
        label: None,
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(swapchain_format.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: swapchain_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: swapchain_capabilities.alpha_modes[0],
        view_formats: vec![],
    };

    surface.configure(&device, &config);
    window.request_redraw();

    let mut current_mouse_pos = None;

    // array of two mouse positions
    let mut i = 0;
    let mut mouse_positions: [PhysicalPosition<f64>; 2] =
        [winit::dpi::PhysicalPosition::default(); 2];
    let mut move_piece = false;

    event_loop.run(move |event, _, control_flow| {
        // Have the closure take ownership of the resources.
        // `event_loop.run` never returns, therefore we must do this to ensure
        // the resources are properly cleaned up.
        let _ = (&instance, &adapter, &shader, &pipeline_layout);

        *control_flow = ControlFlow::Wait;
        match event {
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                // Reconfigure the surface with the new size
                config.width = size.width;
                config.height = size.height;
                surface.configure(&device, &config);
                // On macos the window needs to be redrawn manually after resizing
                window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::MouseInput { button, state, .. },
                ..
            } => {
                if button == MouseButton::Left {
                    if state == winit::event::ElementState::Released {
                        if let Some(mouse_pos) = current_mouse_pos {
                            mouse_positions[i] = mouse_pos;
                            move_piece = i == 1;
                            i = (i + 1) % 2;
                        }
                    }

                    if move_piece {
                        move_piece = false;
                        let window_size = window.inner_size();
                        if let (Some(from), Some(to)) = (
                            mouse_to_square(mouse_positions[0], window_size),
                            mouse_to_square(mouse_positions[1], window_size),
                        ) {
                            if try_player_move(&mut position, from, to) {
                                let _ = make_move(&mut position);
                                let gpu_board = GpuBoard::new(&position);
                                queue.write_buffer(&uniform_buf, 0, gpu_board.as_byte_slice());
                                window.request_redraw();
                            }
                        }
                    }
                }
            }
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                current_mouse_pos = Some(position);
            }
            Event::RedrawRequested(_) => {
                let frame = surface
                    .get_current_texture()
                    .expect("Failed to acquire next swap chain texture");
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                                store: true,
                            },
                        })],
                        depth_stencil_attachment: None,
                    });
                    rpass.set_bind_group(0, &bind_group, &[]);
                    rpass.set_pipeline(&render_pipeline);
                    rpass.draw(0..6, 0..1);
                }

                queue.submit(Some(encoder.finish()));
                frame.present();
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => {}
        }
    });
}

fn main() {
    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("Onion")
        .with_inner_size(winit::dpi::LogicalSize::new(800, 800))
        .build(&event_loop)
        .unwrap();
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run(event_loop, window));
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");

        use winit::platform::web::WindowExtWebSys;
        // On wasm, append the canvas to the document body
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(window.canvas()))
                    .ok()
            })
            .expect("couldn't append canvas to document body");
        wasm_bindgen_futures::spawn_local(run(event_loop, window));
    }
}
