#![forbid(unsafe_code)]

mod initialize;
mod policy;
mod rng;
mod rollout;

pub use flat::{flat_monte_carlo, flat_monte_carlo_until, FlatResult};
pub use initialize::{initialize_base, NumberPlacement};
pub use policy::{choose_action, sample_outcome, Policy};
pub use rng::{derive_seed, SearchRng, StreamKind};
pub use rollout::{rollout, RolloutLimits, RolloutResult, RolloutScratch};
mod flat;
