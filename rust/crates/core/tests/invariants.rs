use std::mem::size_of;

use catanatron_core::{apply_checked, Status};
use catanatron_core::{
    edge_endpoints, generate_actions, node_neighbors, BASE_EDGE_COUNT, BASE_NODE_COUNT,
};
use catanatron_core::{
    validate_boundary, validate_outcome, Action, ChanceKind, DevelopmentCard, EdgeId,
    IllegalAction, NodeId, Outcome, Phase, PlayerId, Position, Resource, TileId,
};

#[test]
fn ids_reject_out_of_range_values() {
    assert!(PlayerId::new(4).is_err());
    assert!(NodeId::new(54).is_err());
    assert!(EdgeId::new(72).is_err());
    assert!(TileId::new(19).is_err());
}

#[test]
fn exported_base_topology_has_expected_dense_bounds() {
    assert_eq!(BASE_NODE_COUNT, 54);
    assert_eq!(BASE_EDGE_COUNT, 72);
    let endpoints = edge_endpoints(EdgeId::new(0).unwrap());
    assert_eq!((endpoints.0.get(), endpoints.1.get()), (0, 1));
    assert_eq!(node_neighbors(NodeId::new(0).unwrap()).count(), 3);
}

#[test]
fn setup_generation_respects_distance_and_remembers_settlement_for_road() {
    let mut position = Position::new(2).unwrap();
    let mut actions = Vec::new();
    generate_actions(&position, &mut actions);
    assert_eq!(actions.len(), 54);
    let settlement = NodeId::new(0).unwrap();
    position.buildings[0] = 1;
    position.phase = Phase::SetupRoad {
        actor: position.actor,
        settlement,
        reverse: false,
    };
    generate_actions(&position, &mut actions);
    assert_eq!(actions.len(), 3);
    assert!(actions.iter().all(|action| matches!(action, Action::BuildRoad(edge) if catanatron_core::incident(*edge, settlement))));
}

#[test]
fn boundary_rejects_wrong_actor_phase_chance_and_terminal() {
    let mut position = Position::new(2).unwrap();
    let player_one = PlayerId::new(1).unwrap();
    assert_eq!(
        validate_boundary(&position, player_one, Action::Roll),
        Err(IllegalAction::WrongActor {
            expected: position.actor,
            actual: player_one
        })
    );
    assert_eq!(
        validate_boundary(&position, position.actor, Action::Roll),
        Err(IllegalAction::WrongPhase)
    );
    position.phase = Phase::PreRoll {
        actor: position.actor,
    };
    assert!(validate_boundary(&position, position.actor, Action::Roll).is_ok());
    position.phase = Phase::Chance {
        actor: position.actor,
        kind: ChanceKind::Dice,
    };
    assert_eq!(
        validate_boundary(&position, position.actor, Action::Roll),
        Err(IllegalAction::WrongPhase)
    );
    position.phase = Phase::Terminal;
    assert_eq!(
        validate_boundary(&position, position.actor, Action::Roll),
        Err(IllegalAction::Terminal)
    );
}

#[test]
fn chance_outcomes_must_match_pending_phase_and_available_cards() {
    let mut position = Position::new(2).unwrap();
    position.phase = Phase::Chance {
        actor: position.actor,
        kind: ChanceKind::Dice,
    };
    assert!(validate_outcome(
        &position,
        Outcome::Dice {
            first: 6,
            second: 1
        }
    )
    .is_ok());
    assert_eq!(
        validate_outcome(
            &position,
            Outcome::Dice {
                first: 0,
                second: 7
            }
        ),
        Err(IllegalAction::InvalidOutcome)
    );
    position.phase = Phase::Chance {
        actor: position.actor,
        kind: ChanceKind::Theft {
            victim: PlayerId::new(1).unwrap(),
        },
    };
    position.players[1].hand[Resource::Ore.index()] = 1;
    assert!(validate_outcome(&position, Outcome::StolenResource(Resource::Ore)).is_ok());
    position.phase = Phase::Chance {
        actor: position.actor,
        kind: ChanceKind::DevelopmentCard,
    };
    position.dev_bank[DevelopmentCard::Knight.index()] = 0;
    assert_eq!(
        validate_outcome(&position, Outcome::DevelopmentCard(DevelopmentCard::Knight)),
        Err(IllegalAction::InvalidOutcome)
    );
}

#[test]
fn checked_transitions_are_atomic_and_record_pending_chance() {
    let mut position = Position::new(2).unwrap();
    let actor = position.actor;
    let original = position;
    assert_eq!(
        apply_checked(&mut position, actor, Action::EndTurn),
        Err(IllegalAction::WrongPhase)
    );
    assert_eq!(position, original);
    position.phase = Phase::PreRoll {
        actor: position.actor,
    };
    assert_eq!(
        apply_checked(&mut position, actor, Action::Roll)
            .unwrap()
            .status,
        Status::Chance
    );
    assert!(matches!(
        position.phase,
        Phase::Chance {
            kind: ChanceKind::Dice,
            ..
        }
    ));
    let after_roll = position;
    assert_eq!(
        apply_checked(&mut position, actor, Action::Roll),
        Err(IllegalAction::WrongPhase)
    );
    assert_eq!(position, after_roll);
}

#[test]
fn ending_turn_advances_active_actor_and_clears_dev_flag() {
    let mut position = Position::new(2).unwrap();
    let actor = position.actor;
    position.phase = Phase::PostRoll {
        actor: position.actor,
    };
    position.players[0].played_dev = true;
    assert_eq!(
        apply_checked(&mut position, actor, Action::EndTurn)
            .unwrap()
            .status,
        Status::Decision
    );
    assert_eq!(position.actor, PlayerId::new(1).unwrap());
    assert_eq!(position.turns, 1);
    assert!(!position.players[0].played_dev);
    assert!(matches!(position.phase, Phase::PreRoll { actor } if actor == position.actor));
}

#[test]
fn copy_is_independent_and_uses_no_heap_owned_fields() {
    let root = Position::new(4).unwrap();
    let mut child = root;
    child.players[0].hand[0] = 3;
    child.bank[0] = 16;
    child.buildings[0] = 1;
    assert_eq!(root.players[0].hand[0], 0);
    assert_eq!(root.bank[0], 19);
    assert_eq!(root.buildings[0], 0);
}

#[test]
fn records_current_compact_layout_budget() {
    assert!(
        size_of::<Action>() <= 64,
        "Action unexpectedly grew to {} B",
        size_of::<Action>()
    );
    assert!(
        size_of::<Position>() <= 1024,
        "Position unexpectedly grew to {} B",
        size_of::<Position>()
    );
    eprintln!(
        "Position={}B Action={}B",
        size_of::<Position>(),
        size_of::<Action>()
    );
}
