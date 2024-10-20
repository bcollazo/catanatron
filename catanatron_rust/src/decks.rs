use crate::enums::{DevCard, Resource};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResourceDeck {
    freqdeck: u32, // Store resource counts using 32 bits. Each resource gets 6 bits.
}

impl Default for ResourceDeck {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceDeck {
    /// Static constructor: Constructs a deck with all resources set to 0.
    pub fn new() -> Self {
        ResourceDeck { freqdeck: 0 }
    }

    // Read operations =====
    /// Counts the number of a specific resource in the deck.
    pub fn count(&self, card: Resource) -> u32 {
        let shift = ResourceDeck::resource_shift(card);
        (self.freqdeck >> shift) & 0b111111 // Extract 6 bits (max count 63)
    }

    /// Checks if the deck can draw a specific number of a given resource.
    pub fn can_draw(&self, amount: u32, card: Resource) -> bool {
        let count = self.count(card);
        count >= amount
    }

    // Write operations =====
    /// Draws a specific number of a given resource from the deck.
    pub fn draw(&mut self, amount: u32, card: Resource) {
        let count = self.count(card);
        let new_count = count.saturating_sub(amount); // Avoid negative values
        self.set_resource_count(card, new_count);
    }

    /// Replenishes the deck with a specific number of a given resource.
    pub fn replenish(&mut self, amount: u32, card: Resource) {
        let count = self.count(card);
        let new_count = count.saturating_add(amount); // Avoid overflow
        self.set_resource_count(card, new_count);
    }

    /// Adds two frequency decks together element-wise.
    pub fn add(&self, other: &ResourceDeck) -> ResourceDeck {
        let mut result = ResourceDeck::new();
        for card in [
            Resource::Wood,
            Resource::Brick,
            Resource::Sheep,
            Resource::Wheat,
            Resource::Ore,
        ] {
            let total = self.count(card) + other.count(card);
            result.set_resource_count(card, total);
        }
        result
    }

    /// Subtracts one frequency deck from another element-wise.
    pub fn subtract(&self, other: &ResourceDeck) -> ResourceDeck {
        let mut result = ResourceDeck::new();
        for card in [
            Resource::Wood,
            Resource::Brick,
            Resource::Sheep,
            Resource::Wheat,
            Resource::Ore,
        ] {
            let diff = self.count(card).saturating_sub(other.count(card));
            result.set_resource_count(card, diff);
        }
        result
    }

    /// Checks if one frequency deck contains all the resources of another deck.
    pub fn contains(&self, other: &ResourceDeck) -> bool {
        for card in [
            Resource::Wood,
            Resource::Brick,
            Resource::Sheep,
            Resource::Wheat,
            Resource::Ore,
        ] {
            if self.count(card) < other.count(card) {
                return false;
            }
        }
        true
    }

    /// Creates a frequency deck from a list of resources.
    pub fn from_listdeck(listdeck: &[Resource]) -> ResourceDeck {
        let mut deck = ResourceDeck::new();
        for &resource in listdeck {
            deck.replenish(1, resource);
        }
        deck
    }

    /// Helper function to get the number of bits to shift for a given resource.
    fn resource_shift(card: Resource) -> u32 {
        match card {
            Resource::Wood => 0,   // First 6 bits
            Resource::Brick => 6,  // Next 6 bits
            Resource::Sheep => 12, // Next 6 bits
            Resource::Wheat => 18, // Next 6 bits
            Resource::Ore => 24,   // Next 6 bits
        }
    }

    /// Sets the count of the specified resource using bitwise operations.
    fn set_resource_count(&mut self, card: Resource, count: u32) {
        let shift = ResourceDeck::resource_shift(card);
        self.freqdeck &= !(0b111111 << shift); // Clear the bits for the resource
        self.freqdeck |= (count & 0b111111) << shift; // Set the new count
    }

    /// Static constructor: Constructs a deck with [19, 19, 19, 19, 19] resources.
    pub fn starting_resource_bank() -> Self {
        ResourceDeck::from_counts([19, 19, 19, 19, 19])
    }

    /// Constructs a deck from a list of resource counts
    /// The input array must have 5 elements, corresponding to [Wood, Brick, Sheep, Wheat, Ore].
    pub fn from_counts(counts: [u32; 5]) -> Self {
        let mut freqdeck = 0;

        // Assign each resource count to its corresponding bitfield using bitwise shifts.
        for (i, &count) in counts.iter().enumerate() {
            // Ensure each count fits within 6 bits (maximum value of 63)
            let count = count & 0b111111; // Mask to fit within 6 bits
            freqdeck |= count << (i * 6); // Shift the count to its correct position in the bitfield
        }

        ResourceDeck { freqdeck }
    }

    pub fn total_cards(&self) -> u32 {
        let mut total = 0;

        // Iterate over all resources (Wood, Brick, Sheep, Wheat, Ore)
        for card in [
            Resource::Wood,
            Resource::Brick,
            Resource::Sheep,
            Resource::Wheat,
            Resource::Ore,
        ] {
            total += self.count(card); // Sum each resource count
        }

        total
    }
}

pub fn starting_dev_listdeck() -> [u8; 25] {
    let mut listdeck: [u8; 25] = [0; 25];
    listdeck[0..14].copy_from_slice(&[DevCard::Knight as u8; 14]);
    listdeck[14..16].copy_from_slice(&[DevCard::YearOfPlenty as u8; 2]);
    listdeck[16..18].copy_from_slice(&[DevCard::RoadBuilding as u8; 2]);
    listdeck[18..20].copy_from_slice(&[DevCard::Monopoly as u8; 2]);
    listdeck[20..25].copy_from_slice(&[DevCard::VictoryPoint as u8; 5]);
    listdeck
}

#[derive(Debug, Clone)]
pub struct DevCardDeck {
    pub listdeck: Vec<DevCard>,
}

impl Default for DevCardDeck {
    fn default() -> Self {
        Self::new()
    }
}

impl DevCardDeck {
    /// Static constructor: Constructs a deck with all development cards set to 0.
    pub fn new() -> Self {
        DevCardDeck {
            listdeck: Vec::new(),
        }
    }

    pub fn starting_deck() -> Self {
        let mut listdeck = Vec::new();
        listdeck.extend(vec![DevCard::Knight; 14]);
        listdeck.extend(vec![DevCard::YearOfPlenty; 2]);
        listdeck.extend(vec![DevCard::RoadBuilding; 2]);
        listdeck.extend(vec![DevCard::Monopoly; 2]);
        listdeck.extend(vec![DevCard::VictoryPoint; 5]);
        DevCardDeck { listdeck }
    }

    /// Draws a specific number of a given development card from the deck.
    pub fn draw(&mut self, amount: u32, card: DevCard) {
        let mut count = amount;
        let mut i = 0;
        while count > 0 && i < self.listdeck.len() {
            if self.listdeck[i] == card {
                self.listdeck.remove(i);
                count -= 1;
            } else {
                i += 1;
            }
        }
    }

    /// Replenishes the deck with a specific number of a given development card.
    pub fn replenish(&mut self, amount: u32, card: DevCard) {
        for _ in 0..amount {
            self.listdeck.push(card);
        }
    }

    /// Counts the number of a specific development card in the deck.
    pub fn count(&self, card: DevCard) -> u32 {
        self.listdeck.iter().filter(|&&c| c == card).count() as u32
    }

    /// Returns the total number of development cards in the deck.
    pub fn total_cards(&self) -> u32 {
        self.listdeck.len() as u32
    }

    /// Creates a development card deck from a list of development cards.
    pub fn from_listdeck(listdeck: &[DevCard]) -> DevCardDeck {
        let mut deck = DevCardDeck::new();
        for &card in listdeck {
            deck.replenish(1, card);
        }
        deck
    }

    /// Checks if the deck can draw a specific number of a given development card.
    pub fn can_draw(&self, amount: u32, card: DevCard) -> bool {
        let count = self.count(card);
        count >= amount
    }

    /// Adds two development card decks together element-wise.
    pub fn add(&self, other: &DevCardDeck) -> DevCardDeck {
        let mut result = DevCardDeck::new();
        for card in [
            DevCard::Knight,
            DevCard::VictoryPoint,
            DevCard::RoadBuilding,
            DevCard::YearOfPlenty,
            DevCard::Monopoly,
        ] {
            let total = self.count(card) + other.count(card);
            result.replenish(total, card);
        }
        result
    }

    /// Subtracts one development card deck from another
    pub fn subtract(&self, other: &DevCardDeck) -> DevCardDeck {
        let mut result = DevCardDeck::new();
        for card in [
            DevCard::Knight,
            DevCard::VictoryPoint,
            DevCard::RoadBuilding,
            DevCard::YearOfPlenty,
            DevCard::Monopoly,
        ] {
            let diff = self.count(card).saturating_sub(other.count(card));
            result.replenish(diff, card);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_freqdeck_init() {
        // Test that the starting resource bank is initialized correctly
        let deck = ResourceDeck::starting_resource_bank();
        assert_eq!(deck.count(Resource::Wood), 19);
    }

    #[test]
    fn test_resource_freqdeck_can_draw() {
        // Test the `can_draw` function
        let deck = ResourceDeck::starting_resource_bank();
        assert!(deck.can_draw(10, Resource::Brick)); // Should be able to draw 10 bricks
        assert!(!deck.can_draw(20, Resource::Brick)); // Should not be able to draw 20 bricks
    }

    #[test]
    fn test_resource_freqdeck_integration() {
        let mut deck = ResourceDeck::starting_resource_bank();

        // Test the initial count and total of all resources
        assert_eq!(deck.count(Resource::Wood), 19);
        assert_eq!(deck.count(Resource::Brick), 19);
        assert_eq!(deck.count(Resource::Sheep), 19);
        assert_eq!(deck.count(Resource::Wheat), 19);
        assert_eq!(deck.count(Resource::Ore), 19);

        // Test drawing from the deck
        assert!(deck.can_draw(10, Resource::Wheat));
        deck.draw(10, Resource::Wheat);
        assert_eq!(deck.count(Resource::Wheat), 9); // After drawing 10, there should be 9 Wheat
        assert_eq!(deck.total_cards(), 19 * 5 - 10); // Total sum of resources should be 85

        // Test replenishing the deck
        deck.replenish(5, Resource::Wheat);
        assert_eq!(deck.count(Resource::Wheat), 14); // After replenishing 5, there should be 14 Wheat
        assert_eq!(deck.total_cards(), 19 * 5 - 10 + 5); // Total sum of resources should be 90
    }

    #[test]
    fn test_can_add() {
        let mut a = ResourceDeck::new();
        let mut b = ResourceDeck::new();

        a.replenish(10, Resource::Ore);
        b.replenish(1, Resource::Ore);

        assert_eq!(a.count(Resource::Ore), 10);
        assert_eq!(b.count(Resource::Ore), 1);

        b = b.add(&a);
        assert_eq!(a.count(Resource::Ore), 10);
        assert_eq!(b.count(Resource::Ore), 11);
    }

    #[test]
    fn test_can_subtract() {
        let mut a = ResourceDeck::new();
        let mut b = ResourceDeck::new();

        a.replenish(13, Resource::Sheep);
        b.replenish(4, Resource::Sheep);

        assert_eq!(a.count(Resource::Sheep), 13);
        assert_eq!(b.count(Resource::Sheep), 4);

        b.replenish(11, Resource::Sheep); // now has 15
        b = b.subtract(&a);
        assert_eq!(a.count(Resource::Sheep), 13);
        assert_eq!(b.count(Resource::Sheep), 2);
    }

    #[test]
    fn test_from_array() {
        let a = ResourceDeck::from_listdeck(&[Resource::Brick, Resource::Brick, Resource::Wood]);
        assert_eq!(a.total_cards(), 3);
        assert_eq!(a.count(Resource::Brick), 2);
        assert_eq!(a.count(Resource::Wood), 1);
    }

    #[test]
    fn test_devcard_freqdeck_init() {
        let deck = DevCardDeck::starting_deck();
        assert_eq!(deck.count(DevCard::Knight), 14);
        assert_eq!(deck.count(DevCard::VictoryPoint), 5);
        assert_eq!(deck.total_cards(), 25);
    }

    #[test]
    fn test_devcard_can_draw() {
        let deck = DevCardDeck::starting_deck();
        assert!(deck.can_draw(10, DevCard::Knight));
        assert!(!deck.can_draw(15, DevCard::Knight));
    }

    #[test]
    fn test_devcard_draw() {
        let mut deck = DevCardDeck::starting_deck();
        assert_eq!(deck.count(DevCard::Knight), 14);
        deck.draw(5, DevCard::Knight);
        assert_eq!(deck.count(DevCard::Knight), 9);
        assert_eq!(deck.total_cards(), 25 - 5);
    }
}
