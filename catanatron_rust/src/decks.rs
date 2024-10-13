use super::enums::Resource;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Deck {
    freqdeck: u32, // Store resource counts using 32 bits. Each resource gets 6 bits.
}

impl Deck {
    /// Static constructor: Constructs a deck with all resources set to 0.
    pub fn new() -> Self {
        Deck { freqdeck: 0 }
    }

    // Read operations =====
    /// Counts the number of a specific resource in the deck.
    pub fn count(&self, card: Resource) -> u32 {
        let shift = Deck::resource_shift(card);
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
    pub fn add(&self, other: &Deck) -> Deck {
        let mut result = Deck::new();
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
    pub fn subtract(&self, other: &Deck) -> Deck {
        let mut result = Deck::new();
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
    pub fn contains(&self, other: &Deck) -> bool {
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
    pub fn from_listdeck(listdeck: &[Resource]) -> Deck {
        let mut deck = Deck::new();
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
        let shift = Deck::resource_shift(card);
        self.freqdeck &= !(0b111111 << shift); // Clear the bits for the resource
        self.freqdeck |= (count & 0b111111) << shift; // Set the new count
    }

    /// Static constructor: Constructs a deck with [19, 19, 19, 19, 19] resources.
    pub fn starting_resource_bank() -> Self {
        Deck::from_counts([19, 19, 19, 19, 19])
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

        Deck { freqdeck }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_freqdeck_init() {
        // Test that the starting resource bank is initialized correctly
        let deck = Deck::starting_resource_bank();
        assert_eq!(deck.count(Resource::Wood), 19);
    }

    #[test]
    fn test_resource_freqdeck_can_draw() {
        // Test the `can_draw` function
        let deck = Deck::starting_resource_bank();
        assert!(deck.can_draw(10, Resource::Brick)); // Should be able to draw 10 bricks
        assert!(!deck.can_draw(20, Resource::Brick)); // Should not be able to draw 20 bricks
    }

    #[test]
    fn test_resource_freqdeck_integration() {
        let mut deck = Deck::starting_resource_bank();

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
}
