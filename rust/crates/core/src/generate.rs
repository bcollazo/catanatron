//! Read-only legal move generation for implemented core phases.
use crate::{node_neighbors, Action, EdgeId, Phase, Position, BASE_EDGE_COUNT, BASE_NODE_COUNT};

pub fn generate_actions(position: &Position, out: &mut Vec<Action>) {
    out.clear();
    match position.phase {
        Phase::SetupSettlement { actor, .. } => {
            if position.players[usize::from(actor.get())].pieces[1] == 0 {
                return;
            }
            for raw in 0..BASE_NODE_COUNT as u8 {
                let node = crate::NodeId::new(raw).expect("generated node");
                if position.buildings[usize::from(raw)] == 0
                    && node_neighbors(node)
                        .all(|near| position.buildings[usize::from(near.get())] == 0)
                {
                    out.push(Action::BuildSettlement(node));
                }
            }
        }
        Phase::SetupRoad { settlement, .. } => {
            for raw in 0..BASE_EDGE_COUNT as u8 {
                if position.roads[usize::from(raw)] != 0 {
                    continue;
                }
                let edge = EdgeId::new(raw).expect("generated edge");
                if crate::incident(edge, settlement) {
                    out.push(Action::BuildRoad(edge));
                }
            }
        }
        Phase::PreRoll { actor } => {
            out.push(Action::Roll);
            append_development_actions(position, actor, out);
        }
        Phase::Discard { actor, .. } => {
            for resource in crate::Resource::ALL {
                if position.players[usize::from(actor.get())].hand[resource.index()] > 0 {
                    out.push(Action::Discard(resource));
                }
            }
        }
        Phase::Robber { actor, .. } => {
            for raw in 0..crate::BASE_LAND_TILE_COUNT as u8 {
                if raw == position.robber {
                    continue;
                }
                let tile = crate::TileId::new(raw).expect("generated tile");
                let mut victims = [false; crate::MAX_PLAYERS];
                for node in crate::land_tile_nodes(tile) {
                    let building = position.buildings[usize::from(node.get())];
                    if let Some(owner) = crate::building_owner(building) {
                        if owner != actor
                            && position.players[usize::from(owner.get())]
                                .hand
                                .iter()
                                .any(|&count| count > 0)
                        {
                            victims[usize::from(owner.get())] = true;
                        }
                    }
                }
                let mut found_victim = false;
                for raw_player in 0..position.player_count {
                    if victims[usize::from(raw_player)] {
                        found_victim = true;
                        out.push(Action::MoveRobber {
                            tile,
                            victim: Some(crate::PlayerId::new(raw_player).expect("active player")),
                        });
                    }
                }
                if !found_victim {
                    out.push(Action::MoveRobber { tile, victim: None });
                }
            }
        }
        Phase::FreeRoad { actor, .. } => append_road_placements(position, actor, out),
        Phase::TradeResponse { actor } => {
            out.push(Action::RejectTrade);
            if crate::has_resources(
                &position.players[usize::from(actor.get())].hand,
                &position.trade_receive,
            ) {
                out.push(Action::AcceptTrade);
            }
        }
        Phase::ChooseAccepter { .. } => {
            out.push(Action::CancelTrade);
            for raw in 0..position.player_count {
                if position.trade_accepted_mask & (1 << raw) != 0 {
                    out.push(Action::ConfirmTrade(
                        crate::PlayerId::new(raw).expect("active player"),
                    ));
                }
            }
        }
        Phase::PostRoll { actor } => {
            out.push(Action::EndTurn);
            append_development_actions(position, actor, out);
            let player = &position.players[usize::from(actor.get())];
            if player.pieces[0] > 0 && player.hand[0] > 0 && player.hand[1] > 0 {
                append_road_placements(position, actor, out);
            }
            if player.pieces[1] > 0
                && [0, 1, 2, 3]
                    .into_iter()
                    .all(|resource| player.hand[resource] > 0)
            {
                for raw in 0..BASE_NODE_COUNT as u8 {
                    let node = crate::NodeId::new(raw).expect("generated node");
                    let connected = (0..BASE_EDGE_COUNT as u8).any(|edge| {
                        position.roads[usize::from(edge)] == actor.get() + 1
                            && crate::incident(EdgeId::new(edge).expect("edge"), node)
                    });
                    if connected
                        && position.buildings[usize::from(raw)] == 0
                        && node_neighbors(node)
                            .all(|near| position.buildings[usize::from(near.get())] == 0)
                    {
                        out.push(Action::BuildSettlement(node));
                    }
                }
            }
            if player.pieces[2] > 0
                && player.hand[crate::Resource::Wheat.index()] >= 2
                && player.hand[crate::Resource::Ore.index()] >= 3
            {
                for raw in 0..BASE_NODE_COUNT as u8 {
                    if position.buildings[usize::from(raw)] == actor.get() + 1 {
                        out.push(Action::BuildCity(
                            crate::NodeId::new(raw).expect("generated node"),
                        ));
                    }
                }
            }
            if position.dev_bank.iter().any(|&count| count > 0)
                && player.hand[crate::Resource::Sheep.index()] > 0
                && player.hand[crate::Resource::Wheat.index()] > 0
                && player.hand[crate::Resource::Ore.index()] > 0
            {
                out.push(Action::BuyDevelopmentCard);
            }
        }
        _ => {}
    }
}

pub fn generate_actions_with_context(
    position: &Position,
    context: &crate::GameContext,
    out: &mut Vec<Action>,
) {
    generate_actions(position, out);
    if context.friendly_robber && matches!(position.phase, Phase::Robber { .. }) {
        let unfiltered = out.clone();
        out.retain(|action| {
            let Action::MoveRobber { tile, .. } = action else {
                return true;
            };
            !crate::land_tile_nodes(*tile).into_iter().any(|node| {
                crate::building_owner(position.buildings[usize::from(node.get())]).is_some_and(
                    |owner| {
                        owner != position.actor && crate::actual_victory_points(position, owner) < 3
                    },
                )
            })
        });
        if out.is_empty() {
            out.extend(unfiltered);
        }
        return;
    }
    let Phase::PostRoll { actor } = position.phase else {
        return;
    };
    for give in crate::Resource::ALL {
        let rate = crate::maritime_rate(position, context, actor, give);
        if position.players[usize::from(actor.get())].hand[give.index()] < rate {
            continue;
        }
        for receive in crate::Resource::ALL {
            if receive != give && position.bank[receive.index()] > 0 {
                out.push(Action::MaritimeTrade {
                    give,
                    receive,
                    rate,
                });
            }
        }
    }
}

fn append_development_actions(position: &Position, actor: crate::PlayerId, out: &mut Vec<Action>) {
    let player = &position.players[usize::from(actor.get())];
    if player.played_dev {
        return;
    }
    if player.dev[crate::DevelopmentCard::Knight.index()] > 0
        && player.eligible_dev_mask & (1 << crate::DevelopmentCard::Knight.index()) != 0
    {
        out.push(Action::PlayKnight);
    }
    if player.dev[crate::DevelopmentCard::YearOfPlenty.index()] > 0
        && player.eligible_dev_mask & (1 << crate::DevelopmentCard::YearOfPlenty.index()) != 0
    {
        for first_index in 0..crate::Resource::ALL.len() {
            for second_index in first_index..crate::Resource::ALL.len() {
                let first = crate::Resource::ALL[first_index];
                let second = crate::Resource::ALL[second_index];
                let available = position.bank[first.index()] >= if first == second { 2 } else { 1 }
                    && position.bank[second.index()] >= 1;
                if available {
                    out.push(Action::YearOfPlenty {
                        first,
                        second: Some(second),
                    });
                } else {
                    for resource in [first, second] {
                        let single = Action::YearOfPlenty {
                            first: resource,
                            second: None,
                        };
                        if position.bank[resource.index()] > 0 && !out.contains(&single) {
                            out.push(single);
                        }
                    }
                }
            }
        }
    }
    if player.dev[crate::DevelopmentCard::Monopoly.index()] > 0
        && player.eligible_dev_mask & (1 << crate::DevelopmentCard::Monopoly.index()) != 0
    {
        for resource in crate::Resource::ALL {
            out.push(Action::Monopoly(resource));
        }
    }
    if player.dev[crate::DevelopmentCard::RoadBuilding.index()] > 0
        && player.eligible_dev_mask & (1 << crate::DevelopmentCard::RoadBuilding.index()) != 0
        && player.pieces[0] > 0
    {
        let mut probe = *position;
        probe.phase = Phase::FreeRoad {
            actor,
            remaining: 2,
            resume_post_roll: matches!(position.phase, Phase::PostRoll { .. }),
        };
        if (0..BASE_EDGE_COUNT as u8).any(|raw| {
            crate::validate::validate_road_placement(
                &probe,
                actor,
                EdgeId::new(raw).expect("base edge"),
            )
            .is_ok()
        }) {
            out.push(Action::RoadBuilding);
        }
    }
}

fn append_road_placements(position: &Position, actor: crate::PlayerId, out: &mut Vec<Action>) {
    for raw in 0..BASE_EDGE_COUNT as u8 {
        let edge = EdgeId::new(raw).expect("generated edge");
        if crate::validate::validate_road_placement(position, actor, edge).is_ok() {
            out.push(Action::BuildRoad(edge));
        }
    }
}
