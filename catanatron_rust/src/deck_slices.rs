pub type FreqDeck = [u8; 5];

pub const SETTLEMENT_COST: FreqDeck = [1, 1, 1, 1, 0];
pub const ROAD_COST: FreqDeck = [1, 1, 0, 0, 0];
pub const CITY_COST: FreqDeck = [0, 0, 0, 2, 3];
pub const DEVCARD_COST: FreqDeck = [0, 0, 1, 1, 1];

pub fn freqdeck_sub(freqdeck: &mut [u8], cost: FreqDeck) {
    freqdeck[0] -= cost[0];
    freqdeck[1] -= cost[1];
    freqdeck[2] -= cost[2];
    freqdeck[3] -= cost[3];
    freqdeck[4] -= cost[4];
}

pub fn freqdeck_add(freqdeck: &mut [u8], cost: FreqDeck) {
    freqdeck[0] += cost[0];
    freqdeck[1] += cost[1];
    freqdeck[2] += cost[2];
    freqdeck[3] += cost[3];
    freqdeck[4] += cost[4];
}

pub fn freqdeck_contains(freqdeck: &[u8], cost: FreqDeck) -> bool {
    freqdeck[0] >= cost[0]
        && freqdeck[1] >= cost[1]
        && freqdeck[2] >= cost[2]
        && freqdeck[3] >= cost[3]
        && freqdeck[4] >= cost[4]
}
