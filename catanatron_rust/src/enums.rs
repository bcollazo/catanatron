use crate::map_instance::{EdgeId, NodeId};

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
    Roll(Color),                 // None. Log instead sets it to (int, int) rolled.
    MoveRobber(Color), // value is (coordinate, Color|None). Log has extra element of card stolen.
    Discard(Color),    // value is None|Resource[].
    BuildRoad(u8, EdgeId), // value is edge_id
    BuildSettlement(u8, NodeId), // value is node_id
    BuildCity,         // value is node_id
    BuyDevelopmentCard, // value is None. Log value is card.
    PlayKnightCard,    // value is None
    PlayYearOfPlenty,  // value is (Resource, Resource)
    PlayMonopoly,      // value is Resource
    PlayRoadBuilding,  // value is None
    MaritimeTrade,     // 5-resource tuple, last is resource asked.
    OfferTrade,        // 10-resource tuple, first 5 is offered, last 5 is receiving.
    AcceptTrade,       // 10-resource tuple.
    RejectTrade,       // None
    ConfirmTrade,      // 11-tuple. First 10 like OfferTrade, last is color of accepting player.
    CancelTrade,       // None
    EndTurn,           // None
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
    pub dicard_limit: u8,
    pub vps_to_win: u8,
    pub map_type: MapType,
    pub num_players: u8,
    pub max_ticks: u32,
}
