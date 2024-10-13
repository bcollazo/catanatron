use std::time::Instant;

use enums::Resource;

mod decks;
mod enums;

fn main() {
    // Benchmark deck operations
    println!("Starting benchmark of Deck operations...");
    let start = Instant::now();
    let mut deck = decks::Deck::starting_resource_bank();
    for _ in 0..1_000_000 {
        if deck.can_draw(2, enums::Resource::Wood) {
            deck.draw(2, enums::Resource::Wood);
            deck.replenish(1, Resource::Wood); // Replenish after drawing to keep the count consistent
            deck.replenish(1, Resource::Wood); // Replenish after drawing to keep the count consistent
        }
    }
    let duration = start.elapsed();
    println!("Time taken for 1,000,000 deck operations: {:?}", duration);
    println!("Total cards in deck: {}", deck.total_cards());
}
