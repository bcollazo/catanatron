"""
This is the Python equivalent of one of the Rust benchmarks. 
It shows Rust implementation can be ~600x faster.
"""

import time

from catanatron.models.decks import *


print("Starting benchmark of Deck operations...")

# Get starting time
start = time.time()

# Run benchmark loop for 1,000,000 iterations
for _ in range(1_000_000):
    deck = starting_resource_bank()
    if freqdeck_can_draw(deck, 2, WOOD):
        freqdeck_draw(deck, 2, WOOD)
        freqdeck_replenish(
            deck, 1, WOOD
        )  # Replenish after drawing to keep the count consistent
        freqdeck_replenish(
            deck, 1, WOOD
        )  # Replenish after drawing to keep the count consistent

# Get duration
duration = time.time() - start

# Print the result
print(f"Time taken for 1,000,000 can_draw operations: {duration:.4f} seconds")
