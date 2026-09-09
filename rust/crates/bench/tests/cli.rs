use std::process::Command;

#[test]
fn invalid_cli_options_fail_with_an_error() {
    let output = Command::new(env!("CARGO_BIN_EXE_catanatron-bench"))
        .args(["games", "--players", "5"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .starts_with("error:"));
}
