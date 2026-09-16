import time

from catanatron_rust import Batch


count = 256
batch = Batch(count, players=4, seed=7)
indices = list(range(count))
started = time.perf_counter()
for _ in range(100):
    batch.observe_many(indices)
observe_elapsed = time.perf_counter() - started
started = time.perf_counter()
for cycle in range(20):
    batch.reset_many(indices, [cycle * count + index for index in indices])
reset_elapsed = time.perf_counter() - started
started = time.perf_counter()
result = batch.rollout_many(indices, list(range(count)), threads=8)
elapsed = time.perf_counter() - started
print(f"{count / elapsed:,.0f} rollouts/s; truncated={result['truncated'].sum()}")
print(f"{100 * count / observe_elapsed:,.0f} observations/s including Python crossing")
print(f"{20 * count / reset_elapsed:,.0f} resets/s including observations and menus")
