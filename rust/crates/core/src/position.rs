//! Copyable canonical mutable state; immutable topology stays outside this type.
use crate::{Phase, PlayerId};
pub const MAX_PLAYERS: usize = 4;
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
        })
    }
}
