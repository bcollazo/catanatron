//! Copyable canonical mutable state; immutable topology stays outside this type.
use crate::{Phase, PlayerId};
pub const MAX_PLAYERS: usize = 4;
pub const CITY_OFFSET: u8 = MAX_PLAYERS as u8;

pub const fn building_belongs_to(building: u8, player: PlayerId) -> bool {
    building == player.get() + 1 || building == player.get() + 1 + CITY_OFFSET
}

pub fn building_owner(building: u8) -> Option<PlayerId> {
    let raw = if building == 0 {
        return None;
    } else if building <= MAX_PLAYERS as u8 {
        building - 1
    } else if building <= MAX_PLAYERS as u8 + CITY_OFFSET {
        building - 1 - CITY_OFFSET
    } else {
        return None;
    };
    PlayerId::new(raw).ok()
}

pub const fn building_production(building: u8) -> u8 {
    if building > CITY_OFFSET {
        2
    } else if building > 0 {
        1
    } else {
        0
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlayerState {
    pub hand: [u8; 5],
    pub dev: [u8; 5],
    pub eligible_dev_mask: u8,
    pub played_dev: bool,
    pub pieces: [u8; 3],
    pub played_knights: u8,
}
impl PlayerState {
    pub const EMPTY: Self = Self {
        hand: [0; 5],
        dev: [0; 5],
        eligible_dev_mask: 0,
        played_dev: false,
        pieces: [15, 5, 4],
        played_knights: 0,
    };
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Position {
    pub players: [PlayerState; MAX_PLAYERS],
    pub player_count: u8,
    pub buildings: [u8; 54],
    pub roads: [u8; 72],
    pub bank: [u8; 5],
    pub dev_bank: [u8; 5],
    pub robber: u8,
    pub actor: PlayerId,
    pub turn_owner: PlayerId,
    pub phase: Phase,
    pub turns: u16,
    pub trade_give: [u8; 5],
    pub trade_receive: [u8; 5],
    pub trade_proposer: PlayerId,
    pub trade_responded_mask: u8,
    pub trade_accepted_mask: u8,
    pub longest_road_lengths: [u8; MAX_PLAYERS],
    pub longest_road_holder: Option<PlayerId>,
    pub largest_army_holder: Option<PlayerId>,
}
impl Position {
    pub fn new(player_count: u8) -> Result<Self, crate::IllegalAction> {
        if !(2..=4).contains(&player_count) {
            return Err(crate::IllegalAction::InvalidPlayerCount(player_count));
        }
        let zero = PlayerId::new(0).expect("zero player id");
        Ok(Self {
            players: [PlayerState::EMPTY; MAX_PLAYERS],
            player_count,
            buildings: [0; 54],
            roads: [0; 72],
            bank: [19; 5],
            dev_bank: [14, 2, 2, 2, 5],
            robber: 0,
            actor: zero,
            turn_owner: zero,
            phase: Phase::SetupSettlement {
                actor: zero,
                reverse: false,
            },
            turns: 0,
            trade_give: [0; 5],
            trade_receive: [0; 5],
            trade_proposer: zero,
            trade_responded_mask: 0,
            trade_accepted_mask: 0,
            longest_road_lengths: [0; MAX_PLAYERS],
            longest_road_holder: None,
            largest_army_holder: None,
        })
    }
}
