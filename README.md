# LSF

LSF (Level Set Forecaster) estimates a distribution given only a point estimate (from a NN, boosted tree, or any other model). The empirical distribution is typically used as an uncertainty estimator.

It roughly works like this:
1. Track/store historical (pred, target) data.
2. Given a new prediction, form an empirical distribution of targets by collecting the k-nearest historical predictions and their corresponding true targets.
3. Use the empirical distribution as an uncertainty estimate (calculate quantiles, sample from it, ...).

This implementation is inspired by Level Set Forecaster [1][2]. Unlike in [1], where bins are expanded only to higher bins, this LSF implementation expands a bin in lower and higher directions, collecting the nearest samples for each expansion bin.

### Usage

```python
import numpy as np

N = 10_000
preds = np.random.random(N)
targets = np.random.normal(preds, 0.1)

print("\nBuilding")
lsf = LSF.build(
    preds=preds,
    targets=targets,
    min_bin_size=100,
)

print("\nChecking preds:")
for pred in (-1.0, 0.1, 0.5, 0.9, 1.0, 10.0):
    stats = lsf.stats(pred)
    print("\nPred:", pred)
    print(stats)
    print("sample:", stats.sample(10))
```


### Refs
[1] https://proceedings.neurips.cc/paper_files/paper/2021/file/32b127307a606effdcc8e51f60a45922-Paper.pdf  
[2] https://assets.amazon.science/17/31/0d99831d4add92e711a56abc75f8/forecasting-with-trees.pdf
