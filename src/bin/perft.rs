use onion::engine::{perft, Position};
use std::{env, process, time::Instant};

fn print_usage(binary: &str) {
    eprintln!("usage: {binary} <depth> [initial|<fen>]");
}

fn main() {
    let mut args = env::args();
    let binary = args.next().unwrap_or_else(|| "perft".to_string());

    let depth = match args.next() {
        Some(arg) => match arg.parse::<usize>() {
            Ok(depth) => depth,
            Err(_) => {
                print_usage(&binary);
                process::exit(2);
            }
        },
        None => {
            print_usage(&binary);
            process::exit(2);
        }
    };

    let position = match args.next() {
        None => Position::new(),
        Some(arg) if arg == "initial" => Position::new(),
        Some(fen) => match Position::from_fen(&fen) {
            Ok(position) => position,
            Err(error) => {
                eprintln!("invalid FEN: {error}");
                process::exit(2);
            }
        },
    };

    let start = Instant::now();
    let nodes = perft(&position, depth);
    let seconds = start.elapsed().as_secs_f64();
    let nps = nodes as f64 / seconds.max(1e-9);

    println!("depth {depth}: nodes={nodes} time={seconds:.3}s nps={nps:.0}");
}
