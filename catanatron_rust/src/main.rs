use catanatron_rust::decks;
use catanatron_rust::enums::{Color, GameConfiguration, MapType, Resource, COLORS};
use catanatron_rust::game::play_game;
use catanatron_rust::global_state;
use catanatron_rust::map_instance::MapInstance;
use catanatron_rust::player::{Player, RandomPlayer};
use catanatron_rust::state_vector;
use std::collections::HashMap;
use std::time::Instant;

fn main() {
    // Benchmark deck operations
    println!("Starting benchmark of Deck operations...");
    let start = Instant::now();
    let mut deck = decks::ResourceDeck::starting_resource_bank();
    for _ in 0..1_000_000 {
        if deck.can_draw(2, Resource::Wood) {
            deck.draw(2, Resource::Wood);
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

    let global_state = global_state::GlobalState::new();
    println!("Global State: {:?}", global_state);

    let size = state_vector::get_state_array_size(2);
    println!("Vector length: {}", size);

    let vector = state_vector::initialize_state(4);
    println!("Vector: {:?}", vector);

    let map_instance = MapInstance::new(
        &global_state.base_map_template,
        &global_state.dice_probas,
        0,
    );
    println!("Map Instance Tiles: {:?}", map_instance.get_tile((0, 0, 0)));
    println!(
        "Map Instance Land Tiles: {:?}",
        map_instance.get_land_tile((1, 0, -1))
    );

    println!("Colors slice: {:?}", state_vector::seating_order_slice(4));
    println!("Colors {:?}", COLORS);

    let config = GameConfiguration {
        dicard_limit: 7,
        vps_to_win: 10,
        map_type: MapType::Base,
        num_players: 2,
        max_ticks: 4,
    };
    let mut players: HashMap<u8, Box<dyn Player>> = HashMap::new();
    players.insert(Color::Red as u8, Box::new(RandomPlayer {}));
    players.insert(Color::Blue as u8, Box::new(RandomPlayer {}));

    let result = play_game(global_state, config, players);
    println!("Game result: {:?}", result);
}
