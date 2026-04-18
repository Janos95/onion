use onion::engine::{
    game_result, move_from_uci, move_to_uci, Color, GameResult, Position, Searcher,
};
use std::env;
use std::io::{self, BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Duration;

const DEFAULT_GAMES: usize = 20;
const DEFAULT_MOVETIME_MS: u64 = 50;
const DEFAULT_STOCKFISH_PATH: &str = "stockfish";
const OPENING_SUITE: &[&str] = &[
    "e2e4 e7e5 g1f3 b8c6",
    "d2d4 d7d5 c2c4 e7e6",
    "c2c4 e7e5 b1c3 g8f6",
    "g1f3 d7d5 d2d4 g8f6",
    "e2e4 c7c5 g1f3 d7d6",
    "d2d4 g8f6 c2c4 g7g6",
];

struct Config {
    games: usize,
    movetime_ms: u64,
    stockfish_elo: Option<u32>,
    stockfish_path: String,
}

struct Scoreboard {
    onion_wins: usize,
    draws: usize,
    stockfish_wins: usize,
}

struct Stockfish {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Scoreboard {
    fn record(&mut self, result: GameResult, onion_is_white: bool) {
        match result {
            GameResult::WhiteWin if onion_is_white => self.onion_wins += 1,
            GameResult::BlackWin if !onion_is_white => self.onion_wins += 1,
            GameResult::Draw => self.draws += 1,
            GameResult::WhiteWin | GameResult::BlackWin => self.stockfish_wins += 1,
            GameResult::Ongoing => unreachable!(),
        }
    }

    fn games(&self) -> usize {
        self.onion_wins + self.draws + self.stockfish_wins
    }

    fn score(&self) -> f64 {
        self.onion_wins as f64 + 0.5 * self.draws as f64
    }

    fn score_rate(&self) -> f64 {
        self.score() / self.games() as f64
    }
}

impl Stockfish {
    fn new(path: &str, elo: Option<u32>) -> io::Result<Self> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| io::Error::other("missing stockfish stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("missing stockfish stdout"))?;
        let mut stockfish = Self {
            _child: child,
            stdin,
            stdout: BufReader::new(stdout),
        };

        stockfish.send("uci")?;
        stockfish.read_until("uciok")?;
        stockfish.send("setoption name Threads value 1")?;
        stockfish.send("setoption name Hash value 16")?;
        stockfish.send("setoption name Move Overhead value 0")?;
        if let Some(elo) = elo {
            stockfish.send("setoption name UCI_LimitStrength value true")?;
            stockfish.send(&format!("setoption name UCI_Elo value {elo}"))?;
        }
        stockfish.send("isready")?;
        stockfish.read_until("readyok")?;
        Ok(stockfish)
    }

    fn new_game(&mut self) -> io::Result<()> {
        self.send("ucinewgame")?;
        self.send("isready")?;
        self.read_until("readyok")?;
        Ok(())
    }

    fn best_move(&mut self, moves: &[String], movetime_ms: u64) -> io::Result<Option<String>> {
        let mut position = String::from("position startpos");
        if !moves.is_empty() {
            position.push_str(" moves ");
            position.push_str(&moves.join(" "));
        }
        self.send(&position)?;
        self.send(&format!("go movetime {movetime_ms}"))?;

        loop {
            let line = self.read_line()?;
            if let Some(rest) = line.strip_prefix("bestmove ") {
                let bestmove = rest.split_whitespace().next().unwrap_or("(none)");
                return Ok((bestmove != "(none)").then(|| bestmove.to_string()));
            }
        }
    }

    fn send(&mut self, command: &str) -> io::Result<()> {
        writeln!(self.stdin, "{command}")?;
        self.stdin.flush()
    }

    fn read_until(&mut self, prefix: &str) -> io::Result<()> {
        loop {
            if self.read_line()?.starts_with(prefix) {
                return Ok(());
            }
        }
    }

    fn read_line(&mut self) -> io::Result<String> {
        let mut line = String::new();
        self.stdout.read_line(&mut line)?;
        Ok(line.trim_end().to_string())
    }
}

fn print_usage(binary: &str) {
    eprintln!("usage: {binary} [games] [movetime-ms] [stockfish-elo|full] [stockfish-path]");
}

fn parse_config() -> Result<Config, String> {
    let mut args = env::args();
    let binary = args.next().unwrap_or_else(|| "stockfish_match".to_string());
    let games = match args.next() {
        Some(arg) => arg
            .parse()
            .map_err(|_| format!("invalid games value: {arg}"))?,
        None => DEFAULT_GAMES,
    };
    let movetime_ms = match args.next() {
        Some(arg) => arg
            .parse()
            .map_err(|_| format!("invalid movetime value: {arg}"))?,
        None => DEFAULT_MOVETIME_MS,
    };
    let stockfish_elo = match args.next() {
        Some(arg) if arg == "full" => None,
        Some(arg) => Some(
            arg.parse()
                .map_err(|_| format!("invalid stockfish elo value: {arg}"))?,
        ),
        None => None,
    };
    let stockfish_path = args
        .next()
        .unwrap_or_else(|| DEFAULT_STOCKFISH_PATH.to_string());

    if args.next().is_some() {
        print_usage(&binary);
        return Err("too many arguments".to_string());
    }

    Ok(Config {
        games,
        movetime_ms,
        stockfish_elo,
        stockfish_path,
    })
}

fn apply_opening(
    position: &mut Position,
    history: &mut Vec<u64>,
    move_list: &mut Vec<String>,
    line: &str,
) {
    for uci in line.split_whitespace() {
        let m = move_from_uci(position, uci).expect("opening move must be legal");
        position.do_move(m);
        move_list.push(uci.to_string());
        history.push(position.history_hash());
    }
}

fn play_game(
    stockfish: &mut Stockfish,
    onion_is_white: bool,
    movetime_ms: u64,
    opening_line: &str,
) -> io::Result<GameResult> {
    let mut position = Position::new();
    let mut history = vec![position.history_hash()];
    let mut move_list = Vec::new();
    let mut onion = Searcher::new();

    apply_opening(&mut position, &mut history, &mut move_list, opening_line);

    loop {
        let result = game_result(&position, &history);
        if result != GameResult::Ongoing {
            return Ok(result);
        }

        let onion_to_move = matches!(
            (onion_is_white, position.side_to_move()),
            (true, Color::White) | (false, Color::Black)
        );

        let next_move = if onion_to_move {
            onion
                .best_move_with_time_budget(&position, &history, Duration::from_millis(movetime_ms))
                .map(|(m, _)| move_to_uci(m))
        } else {
            stockfish.best_move(&move_list, movetime_ms)?
        };

        let Some(uci) = next_move else {
            return Ok(game_result(&position, &history));
        };
        let m = move_from_uci(&position, &uci)
            .ok_or_else(|| io::Error::other(format!("illegal move from engine: {uci}")))?;
        position.do_move(m);
        move_list.push(uci);
        history.push(position.history_hash());
    }
}

fn elo_from_score(score_rate: f64) -> Option<f64> {
    if !(0.0..1.0).contains(&score_rate) {
        return None;
    }
    Some(400.0 * (score_rate / (1.0 - score_rate)).log10())
}

fn ci_from_scoreboard(scoreboard: &Scoreboard) -> Option<(f64, f64)> {
    let n = scoreboard.games() as f64;
    if n < 2.0 {
        return None;
    }

    let mean = scoreboard.score_rate();
    if !(0.0..1.0).contains(&mean) {
        return None;
    }

    let win = scoreboard.onion_wins as f64 / n;
    let draw = scoreboard.draws as f64 / n;
    let second_moment = win + 0.25 * draw;
    let variance = (second_moment - mean * mean).max(0.0);
    let standard_error = (variance / n).sqrt();
    let low = (mean - 1.96 * standard_error).clamp(1e-6, 1.0 - 1e-6);
    let high = (mean + 1.96 * standard_error).clamp(1e-6, 1.0 - 1e-6);
    Some((elo_from_score(low).unwrap(), elo_from_score(high).unwrap()))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_config().map_err(io::Error::other)?;
    let mut scoreboard = Scoreboard {
        onion_wins: 0,
        draws: 0,
        stockfish_wins: 0,
    };
    let label = config
        .stockfish_elo
        .map(|elo| format!("Stockfish@{elo}"))
        .unwrap_or_else(|| "Stockfish(full)".to_string());

    for game_idx in 0..config.games {
        let mut stockfish = Stockfish::new(&config.stockfish_path, config.stockfish_elo)?;
        stockfish.new_game()?;
        let onion_is_white = game_idx % 2 == 0;
        let opening_line = OPENING_SUITE[game_idx % OPENING_SUITE.len()];
        let result = play_game(
            &mut stockfish,
            onion_is_white,
            config.movetime_ms,
            opening_line,
        )?;
        scoreboard.record(result, onion_is_white);

        println!(
            "game {:>3}/{:>3}: onion={} opening=\"{}\" result={:?}",
            game_idx + 1,
            config.games,
            if onion_is_white { "white" } else { "black" },
            opening_line,
            result
        );
    }

    let score_rate = scoreboard.score_rate();
    let elo = elo_from_score(score_rate);
    println!();
    println!("opponent: {label}");
    println!(
        "score: {} / {} (wins={}, draws={}, losses={})",
        scoreboard.score(),
        scoreboard.games(),
        scoreboard.onion_wins,
        scoreboard.draws,
        scoreboard.stockfish_wins
    );
    match elo {
        Some(elo) => println!("estimated Elo difference (Onion - {label}): {elo:.0}"),
        None => println!("estimated Elo difference (Onion - {label}): lower/upper bound only"),
    }
    if let Some((low, high)) = ci_from_scoreboard(&scoreboard) {
        println!("approx 95% CI: [{low:.0}, {high:.0}]");
    }

    Ok(())
}
