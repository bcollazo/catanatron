//! The dependency-free rules kernel for Catanatron's Rust rollout engine.
//!
//! This crate intentionally owns only game rules and immutable game data.
//! Search, JSON protocol handling, benchmarks, and Python bindings will live
//! in crates that depend on this one; this crate must not depend on them.

#![forbid(unsafe_code)]

/// Identifies the engine profile implemented by this workspace.
///
/// The detailed compatibility contract is frozen in
/// `rust/docs/rules-profile.md` during E02.
pub const RULES_PROFILE: &str = "rust-v1";

/// Returns the engine profile name without requiring callers to know its
/// storage representation. This gives the initially-created crate a small,
/// tested public contract while later stages add typed game APIs.
#[must_use]
pub const fn rules_profile() -> &'static str {
    RULES_PROFILE
}

#[cfg(test)]
mod tests {
    use super::{rules_profile, RULES_PROFILE};

    #[test]
    fn exposes_the_frozen_profile_identifier() {
        assert_eq!(rules_profile(), RULES_PROFILE);
    }
}
