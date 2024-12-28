use crate::{
    deck_slices::FreqDeck,
    map_instance::{EdgeId, NodeId},
    map_template::Coordinate,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
    Red = 0,
    Blue = 1,
    Orange = 2,
    White = 3,
}

pub const COLORS: [Color; 4] = [Color::Red, Color::Blue, Color::Orange, Color::White];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Resource {
    Wood,
    Brick,
    Sheep,
    Wheat,
    Ore,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DevCard {
    Knight,
    YearOfPlenty,
    Monopoly,
    RoadBuilding,
    VictoryPoint,
}

#[derive(Debug, Clone, Copy)]
pub enum BuildingType {
    Settlement,
    City,
    Road,
}

#[derive(Debug, Clone, Copy)]
pub enum NodeRef {
    North,
    Northeast,
    Southeast,
    South,
    Southwest,
    Northwest,
}

#[derive(Debug, Clone, Copy)]
pub enum EdgeRef {
    East,
    Southeast,
    Southwest,
    West,
    Northwest,
    Northeast,
}

#[derive(Debug, Clone, Copy)]
pub enum ActionPrompt {
    BuildInitialSettlement,
    BuildInitialRoad,
    PlayTurn,
    Discard,
    MoveRobber,
    DecideTrade,
    DecideAcceptees,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Action {
    // The first value in all these is the color of the player.
    Roll(u8, Option<(u8, u8)>), // None. Log instead sets it to (int, int) rolled.
    MoveRobber(u8, Coordinate, Option<u8>), //  Log has extra element of card stolen.
    Discard(u8), // value is None|Resource[].
    BuildRoad(u8, EdgeId),
    BuildSettlement(u8, NodeId),
    BuildCity(u8, NodeId),
    BuyDevelopmentCard(u8), // value is None. Log value is card.
    PlayKnight(u8),
    PlayYearOfPlenty(u8, (Option<u8>, Option<u8>)),
    PlayMonopoly(u8, u8), // value is Resource
    PlayRoadBuilding(u8),

    // First element of tuples is in, last is out.
    MaritimeTrade(u8, (FreqDeck, u8)),
    OfferTrade(u8, (FreqDeck, FreqDeck)),
    AcceptTrade(u8, (FreqDeck, FreqDeck)),
    RejectTrade(u8),
    ConfirmTrade(u8, (FreqDeck, FreqDeck, u8)), // 11-tuple. First 10 like OfferTrade, last is color of accepting player.
    CancelTrade(u8),

    EndTurn(u8), // None
}

#[derive(Debug)]
pub enum MapType {
    Mini,
    Base,
    Tournament,
}

// TODO: Make immutable and read-only
#[derive(Debug)]
pub struct GameConfiguration {
    pub discard_limit: u8,
    pub vps_to_win: u8,
    pub map_type: MapType,
    pub num_players: u8,
    pub max_ticks: u32,
}
