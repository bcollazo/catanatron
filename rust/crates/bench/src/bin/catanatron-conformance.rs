use std::{
    env,
    fs::File,
    io::{self, BufRead, BufReader},
    process::ExitCode,
};

use catanatron_bench::{check_record, FixtureRecord};
use serde_json::json;

fn main() -> ExitCode {
    match run() {
        Ok(count) => {
            println!("{}", json!({"status": "equal", "cases": count}));
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<u64, String> {
    let paths: Vec<_> = env::args_os().skip(1).collect();
    if paths.is_empty() {
        return check_reader(BufReader::new(io::stdin()), "<stdin>");
    }
    let mut count = 0;
    for path in paths {
        let file =
            File::open(&path).map_err(|error| format!("{}: {error}", path.to_string_lossy()))?;
        count += check_reader(BufReader::new(file), &path.to_string_lossy())?;
    }
    Ok(count)
}

fn check_reader(reader: impl BufRead, source: &str) -> Result<u64, String> {
    let mut count = 0;
    for (index, line) in reader.lines().enumerate() {
        let line = line.map_err(|error| format!("{source}:{}: {error}", index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let record: FixtureRecord = serde_json::from_str(&line)
            .map_err(|error| format!("{source}:{}: invalid JSON fixture: {error}", index + 1))?;
        if let Err(error) = check_record(&record) {
            return Err(serde_json::to_string(&error).expect("conformance errors serialize"));
        }
        count += 1;
    }
    Ok(count)
}
