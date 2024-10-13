use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng; // For random number generation

fn copy_array_benchmark(c: &mut Criterion) {
    // Initialize an array of length 1200
    let array = [0u8; 1200];

    c.bench_function("Copy 1200-length array", |b| {
        b.iter(|| {
            let mut rng = rand::thread_rng();
            // Copy the array (deep copy)
            let mut copied_array = black_box(array);
            // Generate a random index between 0 and 1199
            let random_index = rng.gen_range(0..1200);
            // Generate a random value between 0 and 255 (u8 range)
            let random_value = rng.gen_range(0..=255);
            // Modify the array at the random index
            copied_array[random_index] = random_value;
            black_box(copied_array);
        })
    });
}

fn copy_vector_benchmark(c: &mut Criterion) {
    // Initialize a vector of length 600
    let vector = vec![0u8; 600];

    c.bench_function("Clone 600-length vector", |b| {
        b.iter(|| {
            let mut rng = rand::thread_rng();
            // Clone the vector (deep copy)
            let mut copied_vector = black_box(vector.clone());
            // Generate a random index between 0 and 599
            let random_index = rng.gen_range(0..600);
            // Generate a random value between 0 and 255 (u8 range)
            let random_value = rng.gen_range(0..=255);
            // Modify the vector at the random index
            copied_vector[random_index] = random_value;
            black_box(copied_vector);
        })
    });
}

criterion_group!(benches, copy_array_benchmark, copy_vector_benchmark);
criterion_main!(benches);
