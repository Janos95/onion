use onion::engine::{perft, Position};

fn position_from_fen(fen: &str) -> Position {
    Position::from_fen(fen).expect("FEN should parse")
}

fn assert_perft(position: &Position, expected: &[(usize, u64)]) {
    for &(depth, nodes) in expected {
        assert_eq!(
            perft(position, depth),
            nodes,
            "unexpected perft at depth {depth}"
        );
    }
}

#[test]
fn initial_position_matches_standard_perft() {
    let position = Position::new();

    assert_perft(&position, &[(1, 20), (2, 400), (3, 8_902), (4, 197_281)]);
}

#[test]
fn special_move_position_matches_reference_perft() {
    let position =
        position_from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");

    assert_perft(
        &position,
        &[(1, 48), (2, 2_039), (3, 97_862), (4, 4_085_603)],
    );
}

#[test]
fn rook_endgame_matches_standard_perft() {
    let position = position_from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");

    assert_perft(&position, &[(1, 14), (2, 191), (3, 2_812), (4, 43_238)]);
}

#[test]
fn castling_and_promotion_position_matches_standard_perft() {
    let position =
        position_from_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");

    assert_perft(&position, &[(1, 6), (2, 264), (3, 9_467)]);
}

#[test]
fn tactical_position_matches_standard_perft() {
    let position = position_from_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");

    assert_perft(&position, &[(1, 44), (2, 1_486), (3, 62_379)]);
}

#[test]
fn middlegame_position_matches_standard_perft() {
    let position = position_from_fen(
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    );

    assert_perft(
        &position,
        &[(1, 46), (2, 2_079), (3, 89_890), (4, 3_894_594)],
    );
}
