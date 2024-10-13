// tests/integration_test.rs
use catanatron_rust::add;

#[test]
fn test_integration() {
    assert_eq!(add(10, 5), 15);
}
