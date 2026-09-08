use std::io::{self, BufRead, BufWriter, Write};

use serde::Deserialize;
use serde_json::{json, Value};

mod import;

const PROTOCOL_VERSION: u64 = 1;

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

#[derive(Default)]
struct Bot {
    game_id: Option<String>,
    color: Option<String>,
    static_state: Option<Value>,
    rng: u64,
}

impl Bot {
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
                let mut choices = imported.offered;
                if choices.is_empty() {
                    return Err("decide.playable_actions is empty".to_owned());
                }
                self.rng = self
                    .rng
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let action = choices.swap_remove((self.rng as usize) % choices.len());
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

fn run<R: BufRead, W: Write>(reader: R, mut writer: W) -> Result<(), String> {
    let mut bot = Bot::default();
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
    if let Err(error) = run(io::stdin().lock(), BufWriter::new(io::stdout().lock())) {
        eprintln!("catanatron-bot: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transcript(input: &str) -> Result<String, String> {
        let mut output = Vec::new();
        run(input.as_bytes(), &mut output)?;
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
}
