use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn search_best_move(
    board: Vec<i32>,
    side_to_move: u8,
    castling_rights: u8,
    en_passant_square: i32,
    halfmove_clock: u32,
    position_history_hash_parts: Vec<u32>,
    time_budget_ms: u32,
) -> i32 {
    onion::engine::search_best_move(
        &board,
        side_to_move,
        castling_rights,
        en_passant_square,
        halfmove_clock,
        &position_history_hash_parts,
        time_budget_ms,
    )
}

fn main() {}
