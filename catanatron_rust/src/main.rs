use enums::Resource;
use std::time::Instant;

mod decks;
mod enums;
mod state;

fn main() {
    // Benchmark deck operations
    println!("Starting benchmark of Deck operations...");
    let start = Instant::now();
    let mut deck = decks::ResourceDeck::starting_resource_bank();
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

    // Benchmark copy operations
    println!("Starting benchmark of Deck operations...");
    let start = Instant::now();
    let vector = vec![0u8; 600];
    let mut copied_vector = vector.clone();
    for i in 0..1_000_000 {
        let index = i % 600;
        copied_vector[index] = index as u8;
    }
    let duration = start.elapsed();
    println!("Time taken for 1,000,000 copy operations: {:?}", duration);
    println!("Copy Results: {:?}, {:?}", vector, copied_vector);

    // Benchmark array copy operations
    println!("Starting benchmark of Array operations...");
    let start = Instant::now();
    let array = [0u8; 1200];
    let mut copied_array = array;
    for i in 0..1_000_000 {
        let index = i % 1200;
        copied_array[index] = index as u8;
    }
    let duration = start.elapsed();
    println!("Time taken for 1,000,000 array operations: {:?}", duration);
    println!("Copy Results: {:?}, {:?}", array, copied_array);

    let size = state::get_state_array_size(2);
    println!("Vector length: {}", size);

    let vector = state::initialize_state();
    println!("Vector: {:?}", vector);
}
