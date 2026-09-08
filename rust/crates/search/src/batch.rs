use catanatron_core::{GameContext, Position};

use crate::{rollout, Policy, RolloutLimits, RolloutResult, RolloutScratch};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BatchError {
    LengthMismatch,
    NoThreads,
}

#[derive(Clone, Debug)]
pub struct Batch {
    context: GameContext,
    roots: Vec<Position>,
}

impl Batch {
    pub fn new(context: GameContext, roots: Vec<Position>) -> Self {
        Self { context, roots }
    }

    pub fn roots(&self) -> &[Position] {
        &self.roots
    }

    pub fn rollout_many(
        &self,
        seeds: &[u64],
        policy: Policy,
        limits: RolloutLimits,
        threads: usize,
    ) -> Result<Vec<RolloutResult>, BatchError> {
        rollout_many(&self.context, &self.roots, seeds, policy, limits, threads)
    }
}

pub fn rollout_many(
    context: &GameContext,
    roots: &[Position],
    seeds: &[u64],
    policy: Policy,
    limits: RolloutLimits,
    threads: usize,
) -> Result<Vec<RolloutResult>, BatchError> {
    if roots.len() != seeds.len() {
        return Err(BatchError::LengthMismatch);
    }
    if threads == 0 {
        return Err(BatchError::NoThreads);
    }
    if roots.is_empty() {
        return Ok(Vec::new());
    }
    let worker_count = threads.min(roots.len());
    let chunk_size = roots.len().div_ceil(worker_count);
    let chunks = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for start in (0..roots.len()).step_by(chunk_size) {
            let end = (start + chunk_size).min(roots.len());
            handles.push(scope.spawn(move || {
                let mut scratch = RolloutScratch::default();
                let mut output = Vec::with_capacity(end - start);
                for index in start..end {
                    output.push(rollout(
                        context,
                        &roots[index],
                        policy,
                        seeds[index],
                        limits,
                        &mut scratch,
                    ));
                }
                (start, output)
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("rollout worker panicked"))
            .collect::<Vec<_>>()
    });
    let mut output = Vec::with_capacity(roots.len());
    for (_, mut chunk) in chunks {
        output.append(&mut chunk);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{initialize_base, NumberPlacement};
    use catanatron_core::{apply_checked_with_context, generate_actions_with_context};

    #[test]
    fn one_two_and_four_threads_match_scalar_results_exactly() {
        let (context, root) = initialize_base(4, NumberPlacement::OfficialSpiral, 19, 0).unwrap();
        let mut advanced = root;
        let mut menu = Vec::new();
        generate_actions_with_context(&advanced, &context, &mut menu);
        let actor = advanced.actor;
        apply_checked_with_context(&mut advanced, &context, actor, menu[0]).unwrap();
        let roots: Vec<Position> = (0..12)
            .map(|index| if index % 2 == 0 { root } else { advanced })
            .collect();
        let original = roots.clone();
        let seeds: Vec<u64> = (100..112).collect();
        let limits = RolloutLimits {
            turn_limit: 100,
            action_limit: 10_000,
        };
        let expected = rollout_many(&context, &roots, &seeds, Policy::Weighted, limits, 1).unwrap();
        for threads in [2, 4] {
            assert_eq!(
                rollout_many(&context, &roots, &seeds, Policy::Weighted, limits, threads).unwrap(),
                expected
            );
        }
        assert_eq!(roots, original);
    }

    #[test]
    fn validates_batch_shape_before_starting_workers() {
        let (context, root) = initialize_base(2, NumberPlacement::OfficialSpiral, 1, 0).unwrap();
        assert_eq!(
            rollout_many(
                &context,
                &[root],
                &[],
                Policy::Random,
                RolloutLimits::default(),
                1
            ),
            Err(BatchError::LengthMismatch)
        );
        assert_eq!(
            rollout_many(
                &context,
                &[root],
                &[1],
                Policy::Random,
                RolloutLimits::default(),
                0
            ),
            Err(BatchError::NoThreads)
        );
    }
}
