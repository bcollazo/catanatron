use catanatron_rust::decks::{DevCardDeck, ResourceDeck};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn copy_vector_benchmark(c: &mut Criterion) {
    // Initialize a vector of length 600
    let vector = vec![0u8; 600];

    c.bench_function("Clone 600-length vector", |b| {
        let mut iteration_number = 0;
        b.iter(|| {
            let mut copied_vector = vector.clone();
            let index = iteration_number % 600;
            let random_value = (iteration_number % 256) as u8;
            copied_vector[index] = random_value;
            black_box(copied_vector);

            iteration_number += 1;
        })
    });
}

fn copy_array_benchmark(c: &mut Criterion) {
    // Initialize an array of length 1200
    let array = [0u8; 600];

    c.bench_function("Copy 600-length array", |b| {
        let mut iteration_number = 0;
        b.iter(|| {
            let mut copied_array = array;
            let index = iteration_number % 600;
            let random_value = (iteration_number % 256) as u8;
            copied_array[index] = random_value;
            black_box(copied_array);

            iteration_number += 1;
        })
    });
}

#[derive(Debug, Clone)]
pub struct State {
    pub bank_resources: ResourceDeck,
    pub bank_development_cards: DevCardDeck,
}

impl State {
    pub fn initialize_state() -> State {
        State {
            bank_resources: ResourceDeck::starting_resource_bank(),
            bank_development_cards: DevCardDeck::starting_deck(),
        }
    }
}

fn copy_struct_benchmark(c: &mut Criterion) {
    let state = State::initialize_state();

    c.bench_function("Clone Struct1", |b| {
        b.iter(|| {
            let copied_struct1 = black_box(state.clone());
            black_box(copied_struct1);
        })
    });
}

criterion_group!(
    benches,
    copy_vector_benchmark,
    copy_array_benchmark,
    copy_struct_benchmark
);
criterion_main!(benches);
