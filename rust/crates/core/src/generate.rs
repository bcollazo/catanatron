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
        Phase::PostRoll { actor } => {
            out.push(Action::EndTurn);
            let player = &position.players[usize::from(actor.get())];
            if player.pieces[0] == 0 || player.hand[0] == 0 || player.hand[1] == 0 {
                return;
            }
            for raw in 0..BASE_EDGE_COUNT as u8 {
                if position.roads[usize::from(raw)] != 0 {
                    continue;
                }
                let edge = EdgeId::new(raw).expect("generated edge");
                let (a, b) = crate::edge_endpoints(edge);
                let reachable = [a, b].into_iter().any(|node| {
                    let building = position.buildings[usize::from(node.get())];
                    building == actor.get() + 1
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
        _ => {}
    }
}
