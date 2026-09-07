//! Mutually exclusive decision and chance phases.
use crate::{DevelopmentCard, PlayerId, Resource};
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    SetupSettlement {
        actor: PlayerId,
        reverse: bool,
    },
    SetupRoad {
        actor: PlayerId,
        settlement: crate::NodeId,
        reverse: bool,
    },
    PreRoll {
        actor: PlayerId,
    },
    PostRoll {
        actor: PlayerId,
    },
    Discard {
        actor: PlayerId,
        remaining: u8,
    },
    Robber {
        actor: PlayerId,
        resume_post_roll: bool,
    },
    FreeRoad {
        actor: PlayerId,
        remaining: u8,
        resume_post_roll: bool,
    },
    TradeResponse {
        actor: PlayerId,
    },
    ChooseAccepter {
        actor: PlayerId,
    },
    Chance {
        actor: PlayerId,
        kind: ChanceKind,
    },
    Terminal,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChanceKind {
    Dice,
    Theft {
        victim: PlayerId,
        resume_post_roll: bool,
    },
    DevelopmentCard,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Outcome {
    Dice { first: u8, second: u8 },
    StolenResource(Resource),
    DevelopmentCard(DevelopmentCard),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Truncation {
    TurnLimit,
    ActionLimit,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Status {
    Decision,
    Chance,
    Won(PlayerId),
    Truncated(Truncation),
}
