use onion::engine::{perft, Position};
use std::{
    env,
    fs::File,
    io::{self, BufRead, BufReader},
    process,
};

fn print_usage(binary: &str) {
    eprintln!("usage: {binary} <depth> [fen]");
    eprintln!("       {binary} <depth> --file <path>");
}

fn parse_position(fen: Option<String>) -> Position {
    match fen {
        Some(fen) => match Position::from_fen(&fen) {
            Ok(position) => position,
            Err(error) => {
                eprintln!("invalid FEN: {error}");
                process::exit(2);
            }
        },
        None => Position::new(),
    }
}

fn open_fen_reader(path: &str) -> Box<dyn BufRead> {
    if path == "-" {
        Box::new(BufReader::new(io::stdin()))
    } else {
        match File::open(path) {
            Ok(file) => Box::new(BufReader::new(file)),
            Err(error) => {
                eprintln!("could not open {path}: {error}");
                process::exit(2);
            }
        }
    }
}

fn run_batch(depth: usize, path: &str) {
    let reader = open_fen_reader(path);

    for (line_number, line) in reader.lines().enumerate() {
        let line_number = line_number + 1;
        let line = match line {
            Ok(line) => line,
            Err(error) => {
                eprintln!("could not read line {line_number} from {path}: {error}");
                process::exit(2);
            }
        };
        let fen = line.trim();
        if fen.is_empty() || fen.starts_with('#') {
            continue;
        }

        let position = match Position::from_fen(fen) {
            Ok(position) => position,
            Err(error) => {
                eprintln!("invalid FEN on line {line_number}: {error}");
                process::exit(2);
            }
        };

        println!("{}\t{}", perft(&position, depth), fen);
    }
}

fn main() {
    let mut args = env::args();
    let binary = args.next().unwrap_or_else(|| "perft".to_string());
    let Some(depth_arg) = args.next() else {
        print_usage(&binary);
        process::exit(2);
    };
    let depth = match depth_arg.parse::<usize>() {
        Ok(depth) => depth,
        Err(_) => {
            print_usage(&binary);
            process::exit(2);
        }
    };

    match args.next() {
        None => {
            let position = parse_position(None);
            println!("{}", perft(&position, depth));
        }
        Some(flag) if flag == "--file" => {
            let Some(path) = args.next() else {
                print_usage(&binary);
                process::exit(2);
            };
            if args.next().is_some() {
                print_usage(&binary);
                process::exit(2);
            }
            run_batch(depth, &path);
        }
        Some(fen) => {
            if args.next().is_some() {
                print_usage(&binary);
                process::exit(2);
            }
            let position = parse_position(Some(fen));
            println!("{}", perft(&position, depth));
        }
    }
}
