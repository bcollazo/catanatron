use std::{
    collections::BTreeSet,
    env,
    fs::File,
    io::{self, BufRead, BufReader},
    process::ExitCode,
};

use catanatron_bench::{check_record, FixtureRecord};
use serde_json::json;

fn main() -> ExitCode {
    match run() {
        Ok((equal, divergent, divergent_games)) => {
            println!(
                "{}",
                json!({"status": "passed", "equal": equal, "divergent": divergent, "divergent_games": divergent_games})
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(u64, u64, usize), String> {
    let mut allow_known = false;
    let mut paths = Vec::new();
    for argument in env::args_os().skip(1) {
        if argument == "--allow-known-divergences" {
            allow_known = true;
        } else {
            paths.push(argument);
        }
    }
    if paths.is_empty() {
        let (counts, games) = check_reader(BufReader::new(io::stdin()), "<stdin>", allow_known)?;
        return Ok((counts.0, counts.1, games.len()));
    }
    let mut counts = (0, 0);
    let mut divergent_games = BTreeSet::new();
    for path in paths {
        let file =
            File::open(&path).map_err(|error| format!("{}: {error}", path.to_string_lossy()))?;
        let (next, games) =
            check_reader(BufReader::new(file), &path.to_string_lossy(), allow_known)?;
        counts.0 += next.0;
        counts.1 += next.1;
        divergent_games.extend(games);
    }
    Ok((counts.0, counts.1, divergent_games.len()))
}

fn check_reader(
    reader: impl BufRead,
    source: &str,
    allow_known: bool,
) -> Result<((u64, u64), BTreeSet<String>), String> {
    let mut counts = (0, 0);
    let mut divergent_games = BTreeSet::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line.map_err(|error| format!("{source}:{}: {error}", index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let record: FixtureRecord = serde_json::from_str(&line)
            .map_err(|error| format!("{source}:{}: invalid JSON fixture: {error}", index + 1))?;
        match check_record(&record) {
            Ok(()) => counts.0 += 1,
            Err(error) if allow_known && error.field.starts_with("divergence:") => {
                counts.1 += 1;
                divergent_games.insert(
                    error
                        .case_id
                        .rsplit_once('-')
                        .map_or(error.case_id.clone(), |(game, _)| game.to_owned()),
                );
            }
            Err(error) => {
                return Err(serde_json::to_string(&error).expect("conformance errors serialize"));
            }
        }
    }
    Ok((counts, divergent_games))
}
