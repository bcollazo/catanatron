use catanatron_rust::decks;
use catanatron_rust::enums::Resource;
use catanatron_rust::state_vector;

#[test]
fn test_integration() {
    let mut deck = decks::ResourceDeck::starting_resource_bank();
    if deck.can_draw(2, Resource::Wood) {
        deck.draw(2, Resource::Wood);
        deck.replenish(1, Resource::Wood); // Replenish after drawing to keep the count consistent
        deck.replenish(1, Resource::Wood); // Replenish after drawing to keep the count consistent
    }
    assert_eq!(deck.total_cards(), 95);

    let vector = state_vector::initialize_state(2);
    let size = state_vector::get_state_array_size(2);
    assert_eq!(size, vector.len());
}
