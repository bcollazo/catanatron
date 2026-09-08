#![forbid(unsafe_code)]

mod initialize;
mod policy;
mod rng;
mod rollout;

pub use flat::{flat_monte_carlo, flat_monte_carlo_until, FlatResult};
pub use initialize::{initialize_base, initialize_mini, NumberPlacement};
pub use policy::{choose_action, sample_outcome, Policy};
pub use rng::{derive_seed, SearchRng, StreamKind};
pub use rollout::{rollout, rollout_until, RolloutLimits, RolloutResult, RolloutScratch};
mod batch;
mod flat;
pub use batch::{rollout_many, Batch, BatchError};
