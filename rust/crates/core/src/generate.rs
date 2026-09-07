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
        Phase::PreRoll { .. } => out.push(Action::Roll),
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
        Phase::PostRoll { actor } => {
            out.push(Action::EndTurn);
            let player = &position.players[usize::from(actor.get())];
            if player.pieces[0] > 0 && player.hand[0] > 0 && player.hand[1] > 0 {
                for raw in 0..BASE_EDGE_COUNT as u8 {
                    if position.roads[usize::from(raw)] != 0 {
                        continue;
                    }
                    let edge = EdgeId::new(raw).expect("generated edge");
                    let (a, b) = crate::edge_endpoints(edge);
                    let reachable = [a, b].into_iter().any(|node| {
                        let building = position.buildings[usize::from(node.get())];
                        crate::building_belongs_to(building, actor)
                            || (building == 0
                                && (0..BASE_EDGE_COUNT as u8).any(|other| {
                                    position.roads[usize::from(other)] == actor.get() + 1
                                        && crate::incident(EdgeId::new(other).expect("edge"), node)
                                }))
                    });
                    if reachable {
                        out.push(Action::BuildRoad(edge));
                    }
                }
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
