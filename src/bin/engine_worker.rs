use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn search_best_move(board: Vec<i32>, side_to_move: u8, time_budget_ms: u32) -> i32 {
    onion::engine::search_best_move_for_board(&board, side_to_move, time_budget_ms)
}

fn main() {}
