//! The dependency-free rules kernel for Catanatron's Rust rollout engine.
//!
//! This crate intentionally owns only game rules and immutable game data.
//! Search, JSON protocol handling, benchmarks, and Python bindings will live
//! in crates that depend on this one; this crate must not depend on them.

#![forbid(unsafe_code)]

mod action;
mod apply;
mod awards;
mod chance;
mod generate;
mod ids;
mod layout;
mod phase;
mod position;
mod topology;
mod validate;

pub use action::{Action, DevelopmentCard, Resource};
pub use apply::{
    apply_checked, apply_checked_with_context, apply_outcome_checked,
    apply_outcome_checked_with_context, Transition,
};
pub use awards::{actual_victory_points, longest_road_length};
pub use chance::{draw_bounded, enumerate_outcomes, RandomSource, WeightedOutcome};
pub use generate::{generate_actions, generate_actions_with_context};
pub use ids::{EdgeId, IdError, NodeId, PlayerId, TileId};
pub use layout::{maritime_rate, GameContext, LandTile, Layout, LayoutError, Port};
pub use phase::{ChanceKind, Outcome, Phase, Status, Truncation};
pub use position::{
    building_belongs_to, building_owner, building_production, PlayerState, Position, CITY_OFFSET,
    MAX_PLAYERS,
};
pub use topology::{
    edge_endpoints, incident, land_tile_nodes, node_neighbors, BASE_EDGE_COUNT,
    BASE_LAND_TILE_COUNT, BASE_NODE_COUNT,
};
pub use validate::{validate_boundary, validate_outcome, IllegalAction};

/// Identifies the engine profile implemented by this workspace.
///
/// The detailed compatibility contract is frozen in
/// `rust/docs/rules-profile.md` during E02.
pub const RULES_PROFILE: &str = "rust-v1";

pub(crate) fn has_resources(hand: &[u8; 5], required: &[u8; 5]) -> bool {
    (0..5).all(|index| hand[index] >= required[index])
}

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
