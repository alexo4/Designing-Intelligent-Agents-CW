Final evaluation: greedy actions, 500 games per opponent per run, sides alternated. Mean over seeds ± 95% CI (t-distribution).

### Final checkpoint

| Experiment | Change | Seeds | Win vs random | Win vs held-out heuristic | Win vs heuristic | Pickups/game | Throws/game | Idle |
|---|---|---|---|---|---|---|---|---|
| v1 | baseline: current rewards and trainer | 5 | 91.2% ± 1.8% | 14.4% ± 4.6% | 15.1% ± 5.3% | 1.7 | 14.2 | 34% |
| mix | opponent mix: 40% self, 30% past, 20% heuristic, 10% random | 5 | 87.2% ± 2.6% | 28.6% ± 9.0% | 30.6% ± 9.8% | 2.2 | 17.1 | 38% |
| idle | idle penalty -0.01 | 5 | 91.5% ± 2.3% | 21.7% ± 3.5% | 23.2% ± 4.2% | 2.0 | 14.2 | 26% |
| v2 | v2 = mix + idle penalty (400 updates) | 3 | 90.5% ± 4.3% | 27.6% ± 19.4% | 25.3% ± 8.1% | 2.1 | 14.3 | 23% |
| v2_long | v2 trained for 1500 updates | 3 | 92.0% ± 3.4% | 30.2% ± 5.5% | 29.7% ± 5.5% | 1.8 | 12.8 | 20% |

### Best checkpoint

| Experiment | Change | Seeds | Win vs random | Win vs held-out heuristic | Win vs heuristic | Pickups/game | Throws/game | Idle |
|---|---|---|---|---|---|---|---|---|
| v1 | baseline: current rewards and trainer | 5 | 88.5% ± 2.5% | 21.4% ± 6.7% | 27.6% ± 8.0% | 1.9 | 13.7 | 31% |
| mix | opponent mix: 40% self, 30% past, 20% heuristic, 10% random | 5 | 87.4% ± 1.5% | 32.2% ± 3.8% | 31.8% ± 5.6% | 2.1 | 17.0 | 39% |
| idle | idle penalty -0.01 | 5 | 89.4% ± 2.8% | 19.0% ± 5.0% | 25.6% ± 5.4% | 1.9 | 13.6 | 25% |
| v2 | v2 = mix + idle penalty (400 updates) | 3 | 85.9% ± 2.1% | 31.4% ± 7.7% | 27.7% ± 12.6% | 2.1 | 14.1 | 25% |
| v2_long | v2 trained for 1500 updates | 3 | 91.9% ± 9.8% | 35.9% ± 6.0% | 35.7% ± 7.1% | 1.9 | 14.1 | 20% |
