use std::mem::size_of;

use catanatron_core::{
    apply_checked, apply_checked_with_context, apply_outcome_checked,
    apply_outcome_checked_with_context, Status,
};
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
    assert_eq!(catanatron_core::BASE_LAND_TILE_COUNT, 19);
    assert_eq!(
        catanatron_core::land_tile_nodes(TileId::new(0).unwrap()).len(),
        6
    );
}

#[test]
fn layout_is_immutable_context_data_and_rejects_invalid_tile_assignments() {
    let mut tiles = [catanatron_core::LandTile::DESERT; catanatron_core::BASE_LAND_TILE_COUNT];
    tiles[0] = catanatron_core::LandTile::producing(Resource::Wood, 8);
    let layout = catanatron_core::Layout::new(tiles).unwrap();
    let context = catanatron_core::GameContext::new(layout);
    assert_eq!(
        context.layout.tile(TileId::new(0).unwrap()),
        catanatron_core::LandTile::producing(Resource::Wood, 8)
    );
    tiles[1] = catanatron_core::LandTile::producing(Resource::Brick, 7);
    assert_eq!(
        catanatron_core::Layout::new(tiles),
        Err(catanatron_core::LayoutError::InvalidTile(1))
    );
}

#[test]
fn context_aware_second_setup_settlement_collects_adjacent_resources() {
    let mut tiles = [catanatron_core::LandTile::DESERT; catanatron_core::BASE_LAND_TILE_COUNT];
    tiles[9] = catanatron_core::LandTile::producing(Resource::Wood, 8);
    let context = catanatron_core::GameContext::new(catanatron_core::Layout::new(tiles).unwrap());
    let mut position = Position::new(2).unwrap();
    let actor = position.actor;
    position.buildings[40] = actor.get() + 1;
    position.players[0].pieces[1] -= 1;
    position.phase = Phase::SetupSettlement {
        actor,
        reverse: true,
    };
    apply_checked_with_context(
        &mut position,
        &context,
        actor,
        Action::BuildSettlement(NodeId::new(0).unwrap()),
    )
    .unwrap();
    assert_eq!(position.players[0].hand[Resource::Wood.index()], 1);
    assert_eq!(position.bank[Resource::Wood.index()], 18);
}

#[test]
fn dice_production_pays_settlements_and_cities_but_skips_short_bank_resources() {
    let mut tiles = [catanatron_core::LandTile::DESERT; catanatron_core::BASE_LAND_TILE_COUNT];
    tiles[9] = catanatron_core::LandTile::producing(Resource::Wood, 8);
    let context = catanatron_core::GameContext::new(catanatron_core::Layout::new(tiles).unwrap());
    let mut position = Position::new(2).unwrap();
    let actor = position.actor;
    position.phase = Phase::Chance {
        actor,
        kind: ChanceKind::Dice,
    };
    position.robber = 18;
    position.buildings[0] = actor.get() + 1;
    position.buildings[1] = PlayerId::new(1).unwrap().get() + 1 + catanatron_core::CITY_OFFSET;
    position.bank[Resource::Wood.index()] = 3;
    apply_outcome_checked_with_context(
        &mut position,
        &context,
        Outcome::Dice {
            first: 3,
            second: 5,
        },
    )
    .unwrap();
    assert_eq!(position.players[0].hand[Resource::Wood.index()], 1);
    assert_eq!(position.players[1].hand[Resource::Wood.index()], 2);
    assert_eq!(position.bank[Resource::Wood.index()], 0);

    position.phase = Phase::Chance {
        actor,
        kind: ChanceKind::Dice,
    };
    position.bank[Resource::Wood.index()] = 2;
    assert_eq!(
        apply_outcome_checked_with_context(
            &mut position,
            &context,
            Outcome::Dice {
                first: 3,
                second: 5
            }
        )
        .unwrap()
        .status,
        Status::Decision
    );
    assert_eq!(position.players[0].hand[Resource::Wood.index()], 1);
    assert_eq!(position.players[1].hand[Resource::Wood.index()], 2);
    assert_eq!(position.bank[Resource::Wood.index()], 2);
}

#[test]
fn seven_runs_discards_in_seat_order_then_enters_robber_phase() {
    let context = catanatron_core::GameContext::new(
        catanatron_core::Layout::new(
            [catanatron_core::LandTile::DESERT; catanatron_core::BASE_LAND_TILE_COUNT],
        )
        .unwrap(),
    );
    let mut position = Position::new(3).unwrap();
    let actor = position.actor;
    position.phase = Phase::Chance {
        actor,
        kind: ChanceKind::Dice,
    };
    position.players[0].hand[Resource::Wood.index()] = 8;
    position.players[1].hand[Resource::Brick.index()] = 10;
    apply_outcome_checked_with_context(
        &mut position,
        &context,
        Outcome::Dice {
            first: 3,
            second: 4,
        },
    )
    .unwrap();
    assert!(
        matches!(position.phase, Phase::Discard { actor: current, remaining: 4 } if current == actor)
    );
    for _ in 0..4 {
        apply_checked(&mut position, actor, Action::Discard(Resource::Wood)).unwrap();
    }
    let second = PlayerId::new(1).unwrap();
    assert!(
        matches!(position.phase, Phase::Discard { actor: current, remaining: 5 } if current == second)
    );
    for _ in 0..5 {
        apply_checked(&mut position, second, Action::Discard(Resource::Brick)).unwrap();
    }
    assert!(
        matches!(position.phase, Phase::Robber { actor: current, resume_post_roll: true } if current == actor)
    );
    assert_eq!(position.bank[Resource::Wood.index()], 23);
    assert_eq!(position.bank[Resource::Brick.index()], 24);
}

#[test]
fn robber_moves_to_a_victim_and_resolves_a_checked_theft() {
    let mut position = Position::new(2).unwrap();
    let actor = position.actor;
    let victim = PlayerId::new(1).unwrap();
    position.phase = Phase::Robber {
        actor,
        resume_post_roll: true,
    };
    position.robber = 18;
    position.buildings[0] = victim.get() + 1;
    position.players[1].hand[Resource::Ore.index()] = 1;
    let action = Action::MoveRobber {
        tile: TileId::new(9).unwrap(),
        victim: Some(victim),
    };
    let mut actions = Vec::new();
    generate_actions(&position, &mut actions);
    assert!(actions.contains(&action));
    assert_eq!(
        apply_checked(&mut position, actor, action).unwrap().status,
        Status::Chance
    );
    apply_outcome_checked(&mut position, Outcome::StolenResource(Resource::Ore)).unwrap();
    assert_eq!(position.players[0].hand[Resource::Ore.index()], 1);
    assert_eq!(position.players[1].hand[Resource::Ore.index()], 0);
    assert!(matches!(position.phase, Phase::PostRoll { actor: current } if current == actor));
}

#[test]
fn eligible_knight_can_play_before_roll_and_returns_to_pre_roll_after_theft() {
    let mut position = Position::new(2).unwrap();
    let actor = position.actor;
    let victim = PlayerId::new(1).unwrap();
    position.phase = Phase::PreRoll { actor };
    position.players[0].dev[DevelopmentCard::Knight.index()] = 1;
    position.players[0].eligible_dev_mask = 1;
    position.buildings[0] = victim.get() + 1;
    position.players[1].hand[Resource::Ore.index()] = 1;
    apply_checked(&mut position, actor, Action::PlayKnight).unwrap();
    assert!(matches!(
        position.phase,
        Phase::Robber {
            resume_post_roll: false,
            ..
        }
    ));
    apply_checked(
        &mut position,
        actor,
        Action::MoveRobber {
            tile: TileId::new(9).unwrap(),
            victim: Some(victim),
        },
    )
    .unwrap();
    apply_outcome_checked(&mut position, Outcome::StolenResource(Resource::Ore)).unwrap();
    assert!(matches!(position.phase, Phase::PreRoll { .. }));
    assert!(position.players[0].played_dev);
    assert_eq!(position.players[0].played_knights, 1);
}

#[test]
fn year_of_plenty_and_monopoly_apply_resources_and_card_limit() {
    let mut position = Position::new(3).unwrap();
    let actor = position.actor;
    position.phase = Phase::PostRoll { actor };
    position.players[0].dev[DevelopmentCard::YearOfPlenty.index()] = 1;
    position.players[0].eligible_dev_mask = 1 << DevelopmentCard::YearOfPlenty.index();
    apply_checked(
        &mut position,
        actor,
        Action::YearOfPlenty {
            first: Resource::Wood,
            second: Some(Resource::Wood),
        },
    )
    .unwrap();
    assert_eq!(position.players[0].hand[Resource::Wood.index()], 2);
    assert_eq!(position.bank[Resource::Wood.index()], 17);
    assert!(position.players[0].played_dev);

    position.players[0].played_dev = false;
    position.players[0].dev[DevelopmentCard::Monopoly.index()] = 1;
    position.players[0].eligible_dev_mask = 1 << DevelopmentCard::Monopoly.index();
    position.players[1].hand[Resource::Ore.index()] = 2;
    position.players[2].hand[Resource::Ore.index()] = 3;
    apply_checked(&mut position, actor, Action::Monopoly(Resource::Ore)).unwrap();
    assert_eq!(position.players[0].hand[Resource::Ore.index()], 5);
    assert_eq!(position.players[1].hand[Resource::Ore.index()], 0);
    assert_eq!(position.players[2].hand[Resource::Ore.index()], 0);
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
fn setup_application_follows_two_player_snake_order() {
    let mut position = Position::new(2).unwrap();
    let steps = [(0, 0), (1, 24), (1, 26), (0, 40)];
    for (expected_actor, node) in steps {
        assert_eq!(position.actor.get(), expected_actor);
        let actor = position.actor;
        apply_checked(
            &mut position,
            actor,
            Action::BuildSettlement(NodeId::new(node).unwrap()),
        )
        .unwrap();
        let road = (0..BASE_EDGE_COUNT as u8)
            .map(|raw| EdgeId::new(raw).unwrap())
            .find(|edge| {
                catanatron_core::incident(*edge, NodeId::new(node).unwrap())
                    && position.roads[usize::from(edge.get())] == 0
            })
            .unwrap();
        apply_checked(&mut position, actor, Action::BuildRoad(road)).unwrap();
    }
    assert_eq!(position.actor.get(), 0);
    assert!(matches!(position.phase, Phase::PreRoll { .. }));
    assert_eq!(
        position.turns, 2,
        "only setup roads that advance a seat count"
    );
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
            resume_post_roll: true,
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
fn generated_post_roll_roads_validate_apply_and_pay_atomically() {
    let mut position = Position::new(2).unwrap();
    let actor = position.actor;
    position.phase = Phase::PostRoll { actor };
    position.buildings[0] = actor.get() + 1;
    position.players[0].hand[Resource::Wood.index()] = 1;
    position.players[0].hand[Resource::Brick.index()] = 1;
    position.bank[Resource::Wood.index()] -= 1;
    position.bank[Resource::Brick.index()] -= 1;
    let mut actions = Vec::new();
    generate_actions(&position, &mut actions);
    let road = actions
        .iter()
        .copied()
        .find(|action| matches!(action, Action::BuildRoad(_)))
        .expect("a road incident to the owned building");
    assert!(validate_boundary(&position, actor, road).is_ok());
    apply_checked(&mut position, actor, road).unwrap();
    assert_eq!(position.players[0].hand[Resource::Wood.index()], 0);
    assert_eq!(position.players[0].hand[Resource::Brick.index()], 0);
    assert_eq!(position.bank[Resource::Wood.index()], 19);
    assert_eq!(position.bank[Resource::Brick.index()], 19);
    assert_eq!(position.players[0].pieces[0], 14);
    assert!(matches!(position.phase, Phase::PostRoll { .. }));
}

#[test]
fn post_roll_road_rejects_unaffordable_or_disconnected_placements_without_mutation() {
    let mut position = Position::new(2).unwrap();
    let actor = position.actor;
    position.phase = Phase::PostRoll { actor };
    position.buildings[0] = actor.get() + 1;
    let original = position;
    assert_eq!(
        apply_checked(
            &mut position,
            actor,
            Action::BuildRoad(EdgeId::new(0).unwrap())
        ),
        Err(IllegalAction::InsufficientResource(Resource::Wood))
    );
    assert_eq!(position, original);
    position.players[0].hand[Resource::Wood.index()] = 1;
    position.players[0].hand[Resource::Brick.index()] = 1;
    position.buildings[0] = 0;
    assert_eq!(
        apply_checked(
            &mut position,
            actor,
            Action::BuildRoad(EdgeId::new(0).unwrap())
        ),
        Err(IllegalAction::InvalidRoadPlacement)
    );
    assert_eq!(position.roads, original.roads);
}

#[test]
fn generated_post_roll_settlements_validate_apply_and_pay_atomically() {
    let mut position = Position::new(2).unwrap();
    let actor = position.actor;
    position.phase = Phase::PostRoll { actor };
    position.roads[0] = actor.get() + 1;
    for resource in [
        Resource::Wood,
        Resource::Brick,
        Resource::Sheep,
        Resource::Wheat,
    ] {
        position.players[0].hand[resource.index()] = 1;
        position.bank[resource.index()] -= 1;
    }
    let mut actions = Vec::new();
    generate_actions(&position, &mut actions);
    let settlement = actions
        .iter()
        .copied()
        .find(|action| matches!(action, Action::BuildSettlement(_)))
        .expect("a settlement endpoint on the owned road");
    assert!(validate_boundary(&position, actor, settlement).is_ok());
    apply_checked(&mut position, actor, settlement).unwrap();
    for resource in [
        Resource::Wood,
        Resource::Brick,
        Resource::Sheep,
        Resource::Wheat,
    ] {
        assert_eq!(position.players[0].hand[resource.index()], 0);
        assert_eq!(position.bank[resource.index()], 19);
    }
    assert_eq!(position.players[0].pieces[1], 4);
}

#[test]
fn generated_post_roll_city_upgrades_a_settlement_and_pays_cost() {
    let mut position = Position::new(2).unwrap();
    let actor = position.actor;
    position.phase = Phase::PostRoll { actor };
    position.buildings[0] = actor.get() + 1;
    position.players[0].hand[Resource::Wheat.index()] = 2;
    position.players[0].hand[Resource::Ore.index()] = 3;
    position.bank[Resource::Wheat.index()] -= 2;
    position.bank[Resource::Ore.index()] -= 3;
    let mut actions = Vec::new();
    generate_actions(&position, &mut actions);
    let city = Action::BuildCity(NodeId::new(0).unwrap());
    assert!(actions.contains(&city));
    apply_checked(&mut position, actor, city).unwrap();
    assert_eq!(
        position.buildings[0],
        actor.get() + 1 + catanatron_core::CITY_OFFSET
    );
    assert_eq!(position.players[0].pieces, [15, 6, 3]);
    assert_eq!(position.bank[Resource::Wheat.index()], 19);
    assert_eq!(position.bank[Resource::Ore.index()], 19);
}

#[test]
fn buying_a_development_card_pays_then_resolves_its_draw() {
    let mut position = Position::new(2).unwrap();
    let actor = position.actor;
    position.phase = Phase::PostRoll { actor };
    for resource in [Resource::Sheep, Resource::Wheat, Resource::Ore] {
        position.players[0].hand[resource.index()] = 1;
        position.bank[resource.index()] -= 1;
    }
    let mut actions = Vec::new();
    generate_actions(&position, &mut actions);
    assert!(actions.contains(&Action::BuyDevelopmentCard));
    assert_eq!(
        apply_checked(&mut position, actor, Action::BuyDevelopmentCard)
            .unwrap()
            .status,
        Status::Chance
    );
    for resource in [Resource::Sheep, Resource::Wheat, Resource::Ore] {
        assert_eq!(position.players[0].hand[resource.index()], 0);
        assert_eq!(position.bank[resource.index()], 19);
    }
    assert_eq!(
        apply_outcome_checked(
            &mut position,
            Outcome::DevelopmentCard(DevelopmentCard::Knight)
        )
        .unwrap()
        .status,
        Status::Decision
    );
    assert_eq!(position.players[0].dev[DevelopmentCard::Knight.index()], 1);
    assert_eq!(position.dev_bank[DevelopmentCard::Knight.index()], 13);
    assert!(matches!(position.phase, Phase::PostRoll { actor: current } if current == actor));
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
