Final evaluation: greedy actions, 500 games per opponent per run, sides alternated. Mean over seeds ± 95% CI (t-distribution).

### Final checkpoint

| Experiment | Change | Seeds | Win vs random | Win vs held-out heuristic | Win vs heuristic | Pickups/game | Throws/game | Idle |
|---|---|---|---|---|---|---|---|---|
| v1 | baseline: current rewards and trainer | 5 | 91.2% ± 1.8% | 14.4% ± 4.6% | 15.1% ± 5.3% | 1.7 | 14.2 | 34% |
| mix | opponent mix: 40% self, 30% past, 20% heuristic, 10% random | 5 | 87.2% ± 2.6% | 28.6% ± 9.0% | 30.6% ± 9.8% | 2.2 | 17.1 | 38% |
| dqn_self | DQN self-play, 0.82M steps, buffer 200k | 3 | 88.9% ± 4.9% | 20.2% ± 3.9% | 18.3% ± 5.0% | 2.2 | 11.9 | 49% |
| dqn_mix | DQN + opponent mix (as PPO mix), 0.82M steps, buffer 200k | 3 | 81.7% ± 16.5% | 19.3% ± 8.1% | 25.6% ± 2.6% | 2.0 | 10.8 | 24% |

### Best checkpoint

| Experiment | Change | Seeds | Win vs random | Win vs held-out heuristic | Win vs heuristic | Pickups/game | Throws/game | Idle |
|---|---|---|---|---|---|---|---|---|
| v1 | baseline: current rewards and trainer | 5 | 88.5% ± 2.5% | 21.4% ± 6.7% | 27.6% ± 8.0% | 1.9 | 13.7 | 31% |
| mix | opponent mix: 40% self, 30% past, 20% heuristic, 10% random | 5 | 87.4% ± 1.5% | 32.2% ± 3.8% | 31.8% ± 5.6% | 2.1 | 17.0 | 39% |
| dqn_self | DQN self-play, 0.82M steps, buffer 200k | 3 | 86.5% ± 13.2% | 20.5% ± 12.9% | 24.7% ± 3.8% | 2.0 | 11.4 | 20% |
| dqn_mix | DQN + opponent mix (as PPO mix), 0.82M steps, buffer 200k | 3 | 87.4% ± 8.4% | 17.6% ± 3.9% | 28.9% ± 6.2% | 2.2 | 11.4 | 32% |
