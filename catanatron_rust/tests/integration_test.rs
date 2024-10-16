use catanatron_rust::decks;
use catanatron_rust::enums::Resource;
use catanatron_rust::state;

#[test]
fn test_integration() {
    let mut deck = decks::ResourceDeck::starting_resource_bank();
    if deck.can_draw(2, Resource::Wood) {
        deck.draw(2, Resource::Wood);
        deck.replenish(1, Resource::Wood); // Replenish after drawing to keep the count consistent
        deck.replenish(1, Resource::Wood); // Replenish after drawing to keep the count consistent
    }
    assert_eq!(deck.total_cards(), 95);

    let vector = state::initialize_state();
    let size = state::get_state_array_size(2);
    assert_eq!(size, vector.len());
}
