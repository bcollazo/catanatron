//! Typed player intents. Chance results live in `phase::Outcome`.
use crate::{EdgeId, PlayerId, TileId};
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Resource {
    Wood,
    Brick,
    Sheep,
    Wheat,
    Ore,
}
impl Resource {
    pub const ALL: [Self; 5] = [Self::Wood, Self::Brick, Self::Sheep, Self::Wheat, Self::Ore];
    pub const fn index(self) -> usize {
        self as usize
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum DevelopmentCard {
    Knight,
    YearOfPlenty,
    Monopoly,
    RoadBuilding,
    VictoryPoint,
}
impl DevelopmentCard {
    pub const ALL: [Self; 5] = [
        Self::Knight,
        Self::YearOfPlenty,
        Self::Monopoly,
        Self::RoadBuilding,
        Self::VictoryPoint,
    ];
    pub const fn index(self) -> usize {
        self as usize
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Action {
    Roll,
    EndTurn,
    BuildRoad(EdgeId),
    BuildSettlement(crate::NodeId),
    BuildCity(crate::NodeId),
    BuyDevelopmentCard,
    PlayKnight,
    MoveRobber {
        tile: TileId,
        victim: Option<PlayerId>,
    },
    Discard(Resource),
    YearOfPlenty {
        first: Resource,
        second: Option<Resource>,
    },
    Monopoly(Resource),
    RoadBuilding,
    MaritimeTrade {
        give: Resource,
        receive: Resource,
        rate: u8,
    },
    OfferTrade {
        give: [u8; 5],
        receive: [u8; 5],
    },
    AcceptTrade,
    RejectTrade,
    ConfirmTrade(PlayerId),
    CancelTrade,
}
