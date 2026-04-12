use onion::engine::{create_move, search_root_with_stats, Position};
use std::{env, process, time::Instant};

fn opening_position() -> Position {
    let mut position = Position::new();
    for (from, to) in [(12, 28), (52, 36), (6, 21), (57, 42)] {
        position.do_move(create_move(from, to));
    }
    position
}

fn print_usage(binary: &str) {
    eprintln!("usage: {binary} [initial|opening|all] [max-depth] [max-seconds]");
}

fn benchmark_position(name: &str, position: &Position, max_depth: usize, max_seconds: f64) {
    println!("position: {name}");

    for depth in 1..=max_depth {
        let start = Instant::now();
        let stats = search_root_with_stats(position, depth);
        let elapsed = start.elapsed();
        let seconds = elapsed.as_secs_f64();

        match stats {
            Some(stats) => {
                let nps = stats.nodes as f64 / seconds.max(1e-9);
                println!(
                    "depth {depth}: nodes={} score={} time={seconds:.3}s nps={nps:.0}",
                    stats.nodes, stats.score
                );
            }
            None => {
                println!("depth {depth}: no legal moves");
                break;
            }
        }

        if seconds > max_seconds {
            break;
        }
    }
}

fn main() {
    let mut args = env::args();
    let binary = args.next().unwrap_or_else(|| "bench".to_string());
    let position_name = args.next().unwrap_or_else(|| "all".to_string());
    let max_depth = match args.next() {
        Some(arg) => match arg.parse::<usize>() {
            Ok(depth) => depth,
            Err(_) => {
                print_usage(&binary);
                process::exit(2);
            }
        },
        None => 8,
    };
    let max_seconds = match args.next() {
        Some(arg) => match arg.parse::<f64>() {
            Ok(seconds) => seconds,
            Err(_) => {
                print_usage(&binary);
                process::exit(2);
            }
        },
        None => 3.0,
    };

    let initial = Position::new();
    let opening = opening_position();

    match position_name.as_str() {
        "initial" => benchmark_position("initial", &initial, max_depth, max_seconds),
        "opening" => benchmark_position("opening", &opening, max_depth, max_seconds),
        "all" => {
            benchmark_position("initial", &initial, max_depth, max_seconds);
            benchmark_position("opening", &opening, max_depth, max_seconds);
        }
        _ => {
            print_usage(&binary);
            process::exit(2);
        }
    }
}
