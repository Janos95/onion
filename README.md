# onion

Simple chess engine using an sdf based rendering. 
Play [here](https://janos95.github.io/onion/), but beaware, it might make you cry :)

![SDF chess showcase](sdf_chess.gif)

## Local

- Run the app: `cargo run`
- Run the design lab: `cargo run --bin design_lab`
- Inspect a single piece: `cargo run --bin design_lab -- --piece knight`
- Edit `design_lab_scenes.txt` to hot-reload lab scenarios
- Benchmark the engine: `cargo run --release --bin bench -- opening 8 3.0`
- Run perft from the initial position: `cargo run --release --bin perft -- 4`
- Run perft from a FEN: `cargo run --release --bin perft -- 3 "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPB1PPP/R3K2R w KQkq - 0 1"`
- Run perft for many FENs from a file: `cargo run --release --bin perft -- 3 --file fens.txt`
