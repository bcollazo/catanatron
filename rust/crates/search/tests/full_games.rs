use catanatron_search::{
    initialize_base, rollout, NumberPlacement, Policy, RolloutLimits, RolloutScratch,
};

#[test]
fn completes_seeded_game_matrix_for_two_and_four_players() {
    let limits = RolloutLimits::default();
    for players in [2, 4] {
        for policy in [Policy::Random, Policy::Weighted] {
            let mut scratch = RolloutScratch::default();
            let mut winners = 0;
            let mut turn_limits = 0;
            for game in 0..100 {
                let (context, root) =
                    initialize_base(players, NumberPlacement::OfficialSpiral, 0, game).unwrap();
                let result = rollout(&context, &root, policy, game, limits, &mut scratch);
                assert!(result.winner.is_some() || result.truncation.is_some());
                assert_ne!(
                    result.truncation,
                    Some(catanatron_core::Truncation::ActionLimit)
                );
                assert!(result.player_actions > 0);
                winners += usize::from(result.winner.is_some());
                turn_limits +=
                    usize::from(result.truncation == Some(catanatron_core::Truncation::TurnLimit));
            }
            eprintln!(
                "players={players} policy={policy:?} winners={winners} turn_limits={turn_limits}"
            );
        }
    }
}
