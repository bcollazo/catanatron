import time

from catanatron_rust import Batch


count = 256
batch = Batch(count, players=4, seed=7)
indices = list(range(count))
started = time.perf_counter()
result = batch.rollout_many(indices, list(range(count)), threads=8)
elapsed = time.perf_counter() - started
print(f"{count / elapsed:,.0f} rollouts/s; truncated={result['truncated'].sum()}")
