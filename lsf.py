'''
Uncertainty estimator inspired by Level Set Forecaster [1][2].

Unlike in [1], where bins are expanded only to higher higher bins, this
LSF implementation expands a bin in both the lower and higher directions,
collecting the nearest samples for each expansion bin.

[1] https://proceedings.neurips.cc/paper_files/paper/2021/file/32b127307a606effdcc8e51f60a45922-Paper.pdf
[2] https://assets.amazon.science/17/31/0d99831d4add92e711a56abc75f8/forecasting-with-trees.pdf
'''
from collections import defaultdict
from dataclasses import dataclass
from typing import List
import bisect

import numpy as np

Q = np.array(list(range(101))) # Note: Changing 101 value will affect LSFBinStats


@dataclass
class LSFBinStats:
    size: int
    mean: float
    median: float
    min: float
    max: float
    quantiles: List[float] # [0, 1, 2, ..., 100]

    @classmethod
    def build(Cls, samples):
        samp = np.array(samples)
        quantiles = np.percentile(samp, Q)
        return Cls(
            size=samp.size,
            mean=samp.mean(),
            median=quantiles[50],
            min=quantiles[0],
            max=quantiles[100],
            quantiles=quantiles,
        )

    def sample(self, n: int):
        '''
        Sample values from this bin's empirical distribution
        by interpolating between pre-calculated quantiles.
        '''
        q_rvals = np.random.random(n) * 100.0 # [3.32, 2.10, 0.99]
        q_idxs = np.floor(q_rvals) # [3.0, 2.0, 0.0, ...]
        q_interp = q_rvals - q_idxs # [0.0, 1.0)
        q_idxs = q_idxs.astype(int) # [3, 2, 0, ..]

        # Interpolate between lower/higher quantile bucket
        res = (1.0 - q_interp) * self.quantiles[q_idxs] # lower
        res += q_interp * self.quantiles[q_idxs + 1] # higher
        return res


Pred = float
Target = float


@dataclass
class LSF:
    preds: List[Pred] # Sorted preds
    bins: List[List[Target]] # original outcomes corresponding to :preds
    min_bin_size: int
    bins_expanded: List[List[Target]] # expanded to be >= :min_bin_size
    bin_stats: List[LSFBinStats] # Computed from bins_expanded

    @classmethod
    def build(
        Cls,
        preds: List[float], # or ndarray
        targets: List[float], # or ndarray
        min_bin_size: int = 100,
    ):
        assert len(preds) == len(targets)

        print("\n[Building LSF]")
        print("...Making prediction bins")
        # Build pred -> List[target]
        targets_by_pred = defaultdict(list)
        for i in range(len(preds)):
            targets_by_pred[preds[i]].append(targets[i])

        # Collect bins + preds
        bins, preds = [], []
        for pred in sorted(targets_by_pred.keys()):
            preds.append(pred)
            bins.append(targets_by_pred[pred])

        # Build incomplete object
        c = Cls(
            preds=preds,
            bins=bins,
            min_bin_size=min_bin_size,
            bins_expanded=[],
            bin_stats=[],
        )

        print("...Expanding bins")
        # Compute expanded bins + stats for each bin
        for pred in preds:
            samples = c.empirical(pred, min_bin_size)
            c.bins_expanded.append(samples)

            stats = LSFBinStats.build(samples)
            c.bin_stats.append(stats)
        print("...DONE")

        return c

    def find_nearest(self, pred: Pred):
        '''
        Find bin idx nearest to :pred.

        Tie goes to lower bin.
        '''
        # Find insertion point (not nearest necessarily)
        index = bisect.bisect_left(self.preds, pred)
        if index == 0:
            return 0
        if index == len(self.preds):
            return len(self.preds) - 1

        # Compare the neighboring values
        before = self.preds[index - 1]
        after = self.preds[index]
        if (after - pred) < (pred - before):
            return index
        else:
            return index - 1

    def empirical(self, pred: Pred, min_samples: int):
        '''
        Get the empirical distribution formed by taking at
        least :min_samples samples from bins nearest to :pred.
        '''
        max_idx = len(self.bins) - 1
        start = self.find_nearest(pred)
        queue = [(start, abs(pred - self.preds[start]), True, True)]
        emp = []
        while queue:
            # Get next nearest bin to pred XXX: use pqueue instead
            queue.sort(key=lambda x: x[1], reverse=True)
            bin_idx, _, lower, higher = queue.pop()

            # Add samples from bin to empirical samples
            emp.extend(self.bins[bin_idx])
            if len(emp) >= min_samples:
                break

            # Expand to lower idx
            if lower and (bin_idx > 0):
                queue.append((bin_idx - 1, abs(pred - self.preds[bin_idx - 1]), True, False))

            # Expand to higher idx
            if higher and (bin_idx < max_idx):
                queue.append((bin_idx + 1, abs(pred - self.preds[bin_idx + 1]), False, True))
        return emp

    def stats(self, pred: Pred):
        nbin = self.find_nearest(pred)
        return self.bin_stats[nbin]


if __name__ == "__main__":

    ######################
    # Usage
    ######################
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
