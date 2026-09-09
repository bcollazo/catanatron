use std::{
    env,
    io::{self, BufRead, BufWriter, Write},
    time::{Duration, Instant},
};

use serde::Deserialize;
use serde_json::{json, Value};

mod import;

const PROTOCOL_VERSION: u64 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Policy {
    Random,
    Rollout,
}

#[derive(Clone, Copy, Debug)]
struct Config {
    policy: Policy,
    simulations: u32,
    budget_ms: u64,
    seed: u64,
    threads: u16,
    metrics: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            policy: Policy::Random,
            simulations: 1_000,
            budget_ms: 100,
            seed: 0,
            threads: 1,
            metrics: false,
        }
    }
}

#[derive(Debug, Deserialize)]
struct Envelope {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    protocol_version: Option<u64>,
    #[serde(default)]
    game_id: Option<String>,
    #[serde(default)]
    color: Option<String>,
    #[serde(default)]
    playable_actions: Option<Vec<Value>>,
    #[serde(default)]
    state: Option<Value>,
}

struct Bot {
    game_id: Option<String>,
    color: Option<String>,
    static_state: Option<Value>,
    rng: u64,
    config: Config,
    decisions: u64,
}

impl Default for Bot {
    fn default() -> Self {
        Self::new(Config::default())
    }
}

impl Bot {
    fn new(config: Config) -> Self {
        Self {
            game_id: None,
            color: None,
            static_state: None,
            rng: config.seed,
            config,
            decisions: 0,
        }
    }

    fn handle(&mut self, message: Envelope) -> Result<Option<Value>, String> {
        match message.kind.as_str() {
            "hello" => {
                if message.protocol_version != Some(PROTOCOL_VERSION) {
                    return Err(format!(
                        "unsupported protocol_version {:?}; expected {PROTOCOL_VERSION}",
                        message.protocol_version
                    ));
                }
                Ok(Some(json!({
                    "protocol_version": PROTOCOL_VERSION,
                    "name": "catanatron-rust",
                    "observe": false
                })))
            }
            "before" => {
                let game_id = required(message.game_id, "before.game_id")?;
                let color = required(message.color, "before.color")?;
                self.game_id = Some(game_id);
                self.color = Some(color);
                self.static_state = Some(required(message.state, "before.state")?);
                Ok(None)
            }
            "decide" => {
                let game_id = required(message.game_id, "decide.game_id")?;
                let color = required(message.color, "decide.color")?;
                if self.game_id.as_deref() != Some(game_id.as_str()) {
                    return Err(format!(
                        "decide.game_id {game_id:?} does not match active game"
                    ));
                }
                if self.color.as_deref() != Some(color.as_str()) {
                    return Err(format!("decide.color {color:?} does not match bot color"));
                }
                let actions = required(message.playable_actions, "decide.playable_actions")?;
                for action in &actions {
                    validate_action_shape(action)?;
                }
                let dynamic = required(message.state, "decide.state")?;
                let mut state = dynamic
                    .as_object()
                    .cloned()
                    .ok_or("decide.state must be an object")?;
                let static_state = self
                    .static_state
                    .as_ref()
                    .and_then(Value::as_object)
                    .ok_or("before.state must be an object")?;
                state.insert(
                    "map".to_owned(),
                    static_state
                        .get("map")
                        .ok_or("before.state.map is missing")?
                        .clone(),
                );
                let imported = import::import(Value::Object(state), &game_id, &color, actions)?;
                let _root = (&imported.context, &imported.position);
                let choices = imported.offered;
                if choices.is_empty() {
                    return Err("decide.playable_actions is empty".to_owned());
                }
                let action = if choices.len() == 1 {
                    choices[0].clone()
                } else if self.config.policy == Policy::Rollout {
                    let deadline = Instant::now()
                        + Duration::from_millis(self.config.budget_ms.saturating_sub(5));
                    let typed: Vec<_> = choices.iter().map(|choice| choice.1).collect();
                    let result = catanatron_search::flat_monte_carlo_until(
                        &imported.context,
                        &imported.position,
                        &typed,
                        self.config.simulations,
                        self.config.seed.wrapping_add(self.decisions),
                        catanatron_search::RolloutLimits::default(),
                        || Instant::now() >= deadline,
                    )
                    .ok_or("rollout search received an empty menu")?;
                    if self.config.metrics {
                        eprintln!(
                            "catanatron_search_metrics {{\"rollouts\":{},\"decision\":{}}}",
                            result.total_samples, self.decisions
                        );
                    }
                    choices
                        .iter()
                        .find(|choice| choice.1 == result.action)
                        .cloned()
                        .ok_or("search selected an unoffered action")?
                } else {
                    self.rng = self
                        .rng
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1);
                    choices[(self.rng as usize) % choices.len()].clone()
                };
                self.decisions = self.decisions.wrapping_add(1);
                Ok(Some(json!({"action": action.0})))
            }
            "after" => {
                if let Some(game_id) = message.game_id {
                    if self.game_id.as_deref() != Some(game_id.as_str()) {
                        return Err(format!(
                            "after.game_id {game_id:?} does not match active game"
                        ));
                    }
                }
                self.game_id = None;
                self.color = None;
                self.static_state = None;
                Ok(None)
            }
            // Observation is disabled, but tolerate optional or future notifications.
            _ => Ok(None),
        }
    }
}

fn required<T>(value: Option<T>, field: &str) -> Result<T, String> {
    value.ok_or_else(|| format!("missing {field}"))
}

fn validate_action_shape(action: &Value) -> Result<(), String> {
    let values = action
        .as_array()
        .filter(|values| values.len() == 3)
        .ok_or("offered action must be [color, action_type, value]")?;
    if !values[0].is_string() || !values[1].is_string() {
        return Err("offered action color and action_type must be strings".to_owned());
    }
    Ok(())
}

fn run<R: BufRead, W: Write>(reader: R, mut writer: W, config: Config) -> Result<(), String> {
    let mut bot = Bot::new(config);
    for line in reader.lines() {
        let line = line.map_err(|error| format!("stdin read failed: {error}"))?;
        let message: Envelope = serde_json::from_str(&line)
            .map_err(|error| format!("invalid JSON message: {error}"))?;
        if let Some(reply) = bot.handle(message)? {
            serde_json::to_writer(&mut writer, &reply)
                .map_err(|error| format!("stdout write failed: {error}"))?;
            writer
                .write_all(b"\n")
                .and_then(|_| writer.flush())
                .map_err(|error| format!("stdout flush failed: {error}"))?;
        }
    }
    Ok(())
}

fn main() {
    let result = parse_args(env::args().skip(1)).and_then(|config| {
        run(
            io::stdin().lock(),
            BufWriter::new(io::stdout().lock()),
            config,
        )
    });
    if let Err(error) = result {
        eprintln!("catanatron-bot: {error}");
        std::process::exit(2);
    }
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Config, String> {
    let mut config = Config::default();
    let mut args = args;
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--policy" => {
                config.policy = match value.as_str() {
                    "random" => Policy::Random,
                    "rollout" => Policy::Rollout,
                    _ => return Err("--policy must be random or rollout".to_owned()),
                }
            }
            "--simulations" => config.simulations = positive(&value, &flag)?,
            "--budget-ms" => config.budget_ms = positive(&value, &flag)?,
            "--seed" => {
                config.seed = value
                    .parse()
                    .map_err(|_| "--seed must be a u64".to_owned())?
            }
            "--threads" => {
                config.threads = positive(&value, &flag)?;
                if config.threads != 1 {
                    return Err(
                        "E10 supports --threads 1; parallel search arrives in E11".to_owned()
                    );
                }
            }
            "--metrics" => {
                config.metrics = match value.as_str() {
                    "true" => true,
                    "false" => false,
                    _ => return Err("--metrics must be true or false".to_owned()),
                }
            }
            _ => return Err(format!("unknown option {flag}")),
        }
    }
    Ok(config)
}

fn positive<T: std::str::FromStr + Default + PartialEq>(
    value: &str,
    flag: &str,
) -> Result<T, String> {
    let parsed = value
        .parse()
        .map_err(|_| format!("{flag} has an invalid value"))?;
    if parsed == T::default() {
        return Err(format!("{flag} must be positive"));
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transcript(input: &str) -> Result<String, String> {
        let mut output = Vec::new();
        run(input.as_bytes(), &mut output, Config::default())?;
        String::from_utf8(output).map_err(|error| error.to_string())
    }

    #[test]
    fn replies_only_to_hello_without_a_decision() {
        let input = concat!(
            "{\"type\":\"hello\",\"protocol_version\":1}\n",
            "{\"type\":\"before\",\"game_id\":\"a\",\"color\":\"BLUE\",\"state\":{}}\n",
            "{\"type\":\"future-notification\"}\n",
            "{\"type\":\"after\",\"game_id\":\"a\"}\n",
            "{\"type\":\"before\",\"game_id\":\"b\",\"color\":\"RED\",\"state\":{}}\n"
        );
        let lines: Vec<Value> = transcript(input)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["observe"], false);
    }

    #[test]
    fn eof_is_clean_and_notifications_need_no_reply() {
        assert_eq!(transcript("").unwrap(), "");
        assert_eq!(transcript("{\"type\":\"step\"}\n").unwrap(), "");
    }

    #[test]
    fn rejects_wrong_ids_and_malformed_offers() {
        let wrong_id = concat!(
            "{\"type\":\"before\",\"game_id\":\"a\",\"color\":\"RED\",\"state\":{}}\n",
            "{\"type\":\"decide\",\"game_id\":\"b\",\"color\":\"RED\",\"playable_actions\":[[\"RED\",\"ROLL\",null]]}\n"
        );
        assert!(transcript(wrong_id).unwrap_err().contains("does not match"));

        let malformed = concat!(
            "{\"type\":\"before\",\"game_id\":\"a\",\"color\":\"RED\",\"state\":{}}\n",
            "{\"type\":\"decide\",\"game_id\":\"a\",\"color\":\"RED\",\"playable_actions\":[[\"RED\",\"ROLL\"]]}\n"
        );
        assert!(transcript(malformed)
            .unwrap_err()
            .contains("offered action"));
    }

    #[test]
    fn parses_search_options_and_rejects_e11_threads() {
        let config = parse_args(
            [
                "--policy",
                "rollout",
                "--simulations",
                "24",
                "--budget-ms",
                "80",
                "--seed",
                "7",
                "--threads",
                "1",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(config.policy, Policy::Rollout);
        assert_eq!(config.simulations, 24);
        assert_eq!(config.budget_ms, 80);
        assert_eq!(config.seed, 7);
        assert!(parse_args(["--threads", "2"].into_iter().map(str::to_owned)).is_err());
        assert!(parse_args(["--simulations", "0"].into_iter().map(str::to_owned)).is_err());
    }
}
