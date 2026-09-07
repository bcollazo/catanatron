//! The dependency-free rules kernel for Catanatron's Rust rollout engine.
//!
//! This crate intentionally owns only game rules and immutable game data.
//! Search, JSON protocol handling, benchmarks, and Python bindings will live
//! in crates that depend on this one; this crate must not depend on them.

#![forbid(unsafe_code)]

mod action;
mod apply;
mod generate;
mod ids;
mod phase;
mod position;
mod topology;
mod validate;

pub use action::{Action, DevelopmentCard, Resource};
pub use apply::{apply_checked, Transition};
pub use generate::generate_actions;
pub use ids::{EdgeId, IdError, NodeId, PlayerId, TileId};
pub use phase::{ChanceKind, Outcome, Phase, Status, Truncation};
pub use position::{PlayerState, Position, MAX_PLAYERS};
pub use topology::{edge_endpoints, incident, node_neighbors, BASE_EDGE_COUNT, BASE_NODE_COUNT};
pub use validate::{validate_boundary, validate_outcome, IllegalAction};

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
