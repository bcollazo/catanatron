//! Exact public award and victory-point maintenance.
use crate::{building_owner, EdgeId, PlayerId, Position};

pub fn longest_road_length(position: &Position, player: PlayerId) -> u8 {
    let mut best = 0;
    for raw in 0..position.map.edge_count() as u8 {
        if position.roads[usize::from(raw)] != player.get() + 1 {
            continue;
        }
        let edge = EdgeId::new(raw).expect("base edge");
        let (a, b) = crate::edge_endpoints_on(position.map, edge).expect("active edge");
        let used = 1_u128 << raw;
        best = best.max(1 + trail(position, player, a, used));
        best = best.max(1 + trail(position, player, b, used));
    }
    best
}

fn trail(position: &Position, player: PlayerId, node: crate::NodeId, used: u128) -> u8 {
    if building_owner(position.buildings[usize::from(node.get())]).is_some_and(|p| p != player) {
        return 0;
    }
    let mut best = 0;
    for raw in 0..position.map.edge_count() as u8 {
        let bit = 1_u128 << raw;
        if used & bit != 0 || position.roads[usize::from(raw)] != player.get() + 1 {
            continue;
        }
        let edge = EdgeId::new(raw).expect("base edge");
        let (a, b) = crate::edge_endpoints_on(position.map, edge).expect("active edge");
        let next = if a == node {
            b
        } else if b == node {
            a
        } else {
            continue;
        };
        best = best.max(1 + trail(position, player, next, used | bit));
    }
    best
}

pub(crate) fn refresh_awards(position: &mut Position) {
    for raw in 0..position.player_count {
        let player = PlayerId::new(raw).expect("active player");
        position.longest_road_lengths[usize::from(raw)] = longest_road_length(position, player);
    }
    position.longest_road_holder = select_holder(
        &position.longest_road_lengths,
        position.player_count,
        5,
        position.longest_road_holder,
    );
    let mut armies = [0; crate::MAX_PLAYERS];
    for raw in 0..position.player_count {
        armies[usize::from(raw)] = position.players[usize::from(raw)].played_knights;
    }
    position.largest_army_holder = select_holder(
        &armies,
        position.player_count,
        3,
        position.largest_army_holder,
    );
}

fn select_holder(
    values: &[u8; crate::MAX_PLAYERS],
    player_count: u8,
    threshold: u8,
    incumbent: Option<PlayerId>,
) -> Option<PlayerId> {
    let maximum = values[..usize::from(player_count)]
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    if maximum < threshold {
        return None;
    }
    if let Some(holder) = incumbent {
        if values[usize::from(holder.get())] == maximum {
            return Some(holder);
        }
    }
    let mut leaders = (0..player_count).filter(|&raw| values[usize::from(raw)] == maximum);
    let first = leaders.next()?;
    if leaders.next().is_some() {
        None
    } else {
        PlayerId::new(first).ok()
    }
}

pub fn actual_victory_points(position: &Position, player: PlayerId) -> u8 {
    let mut points = position.players[usize::from(player.get())].dev
        [crate::DevelopmentCard::VictoryPoint.index()];
    for &building in &position.buildings {
        if crate::building_belongs_to(building, player) {
            points += crate::building_production(building);
        }
    }
    points += 2 * u8::from(position.longest_road_holder == Some(player));
    points += 2 * u8::from(position.largest_army_holder == Some(player));
    points
}
