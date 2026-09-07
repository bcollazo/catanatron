use std::mem::size_of;

use catanatron_core::{
    validate_boundary, Action, ChanceKind, EdgeId, IllegalAction, NodeId, Phase, PlayerId,
    Position, TileId,
};

#[test]
fn ids_reject_out_of_range_values() {
    assert!(PlayerId::new(4).is_err());
    assert!(NodeId::new(54).is_err());
    assert!(EdgeId::new(72).is_err());
    assert!(TileId::new(19).is_err());
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
