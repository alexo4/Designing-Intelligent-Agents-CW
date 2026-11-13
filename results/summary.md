Final evaluation: greedy actions, 500 games per opponent per run, sides alternated. Mean over seeds ± 95% CI (t-distribution).

### Final checkpoint

| Experiment | Change | Seeds | Win vs random | Win vs held-out heuristic | Win vs heuristic | Pickups/game | Throws/game | Idle |
|---|---|---|---|---|---|---|---|---|
| v1 | baseline: current rewards and trainer | 5 | 91.2% ± 1.8% | 14.4% ± 4.6% | 15.1% ± 5.3% | 1.7 | 14.2 | 34% |
| timeout_numbers | time-out win/loss on numbers = +1 / -1 | 3 | 89.7% ± 4.5% | 16.3% ± 27.0% | 15.9% ± 21.6% | 2.0 | 14.9 | 37% |
| team_terminal | win/lose reward also to players eliminated earlier | 3 | 88.1% ± 5.0% | 16.0% ± 16.0% | 15.8% ± 8.8% | 1.9 | 11.7 | 26% |
| pickup0 | pickup reward = 0 | 3 | 91.1% ± 7.2% | 14.5% ± 3.3% | 18.7% ± 9.2% | 2.0 | 15.6 | 39% |
| miss0 | miss penalty = 0 | 3 | 89.7% ± 8.8% | 13.1% ± 14.9% | 17.8% ± 21.7% | 1.8 | 15.3 | 40% |
| sparse | only win/lose (hit, eliminated, pickup, miss = 0) | 3 | 78.2% ± 19.4% | 5.3% ± 4.5% | 6.2% ± 7.2% | 2.4 | 32.5 | 35% |
| idle | idle penalty -0.01 | 5 | 91.5% ± 2.3% | 21.7% ± 3.5% | 23.2% ± 4.2% | 2.0 | 14.2 | 26% |
| advnorm | PPO advantage normalisation | 5 | 91.1% ± 3.6% | 18.0% ± 10.7% | 22.4% ± 11.6% | 2.0 | 14.8 | 32% |
| mix | opponent mix: 40% self, 30% past, 20% heuristic, 10% random | 5 | 87.2% ± 2.6% | 28.6% ± 9.0% | 30.6% ± 9.8% | 2.2 | 17.1 | 38% |
| v2 | v2 = mix + idle penalty (400 updates) | 3 | 90.5% ± 4.3% | 27.6% ± 19.4% | 25.3% ± 8.1% | 2.1 | 14.3 | 23% |
| v2_long | v2 trained for 1500 updates | 3 | 92.0% ± 3.4% | 30.2% ± 5.5% | 29.7% ± 5.5% | 1.8 | 12.8 | 20% |
| dqn_self | DQN self-play, 0.82M steps, buffer 200k | 3 | 88.9% ± 4.9% | 20.2% ± 3.9% | 18.3% ± 5.0% | 2.2 | 11.9 | 49% |
| dqn_mix | DQN + opponent mix (as PPO mix), 0.82M steps, buffer 200k | 3 | 81.7% ± 16.5% | 19.3% ± 8.1% | 25.6% ± 2.6% | 2.0 | 10.8 | 24% |
| v1_dqn_mix20k | DQN 20k, opponent mix 50/25/25 (final) | 1 | 86.2% | 14.2% | 20.0% | 2.2 | 11.1 | 29% |
| v1_legacy_dqn20k | DQN 20k episodes, self-play (final) | 1 | 90.6% | 23.6% | 23.4% | 2.2 | 11.9 | 26% |
| v1_legacy_dqn20k_double | Double DQN 20k (final) | 1 | 89.6% | 15.6% | 17.6% | 2.3 | 11.0 | 37% |

### Best checkpoint

| Experiment | Change | Seeds | Win vs random | Win vs held-out heuristic | Win vs heuristic | Pickups/game | Throws/game | Idle |
|---|---|---|---|---|---|---|---|---|
| v1 | baseline: current rewards and trainer | 5 | 88.5% ± 2.5% | 21.4% ± 6.7% | 27.6% ± 8.0% | 1.9 | 13.7 | 31% |
| timeout_numbers | time-out win/loss on numbers = +1 / -1 | 3 | 88.5% ± 6.1% | 13.0% ± 13.4% | 22.1% ± 23.2% | 1.8 | 14.5 | 31% |
| team_terminal | win/lose reward also to players eliminated earlier | 3 | 83.7% ± 6.1% | 18.9% ± 20.9% | 24.3% ± 18.7% | 1.8 | 11.8 | 28% |
| pickup0 | pickup reward = 0 | 3 | 80.3% ± 12.1% | 17.3% ± 18.9% | 24.0% ± 6.8% | 2.4 | 14.0 | 42% |
| miss0 | miss penalty = 0 | 3 | 90.0% ± 9.0% | 14.9% ± 14.7% | 19.9% ± 10.1% | 1.8 | 14.0 | 37% |
| sparse | only win/lose (hit, eliminated, pickup, miss = 0) | 3 | 73.6% ± 11.7% | 7.2% ± 8.0% | 8.3% ± 7.2% | 2.2 | 25.7 | 36% |
| idle | idle penalty -0.01 | 5 | 89.4% ± 2.8% | 19.0% ± 5.0% | 25.6% ± 5.4% | 1.9 | 13.6 | 25% |
| advnorm | PPO advantage normalisation | 5 | 87.8% ± 6.9% | 24.3% ± 7.8% | 28.3% ± 6.2% | 2.1 | 14.3 | 36% |
| mix | opponent mix: 40% self, 30% past, 20% heuristic, 10% random | 5 | 87.4% ± 1.5% | 32.2% ± 3.8% | 31.8% ± 5.6% | 2.1 | 17.0 | 39% |
| v2 | v2 = mix + idle penalty (400 updates) | 3 | 85.9% ± 2.1% | 31.4% ± 7.7% | 27.7% ± 12.6% | 2.1 | 14.1 | 25% |
| v2_long | v2 trained for 1500 updates | 3 | 91.9% ± 9.8% | 35.9% ± 6.0% | 35.7% ± 7.1% | 1.9 | 14.1 | 20% |
| dqn_self | DQN self-play, 0.82M steps, buffer 200k | 3 | 86.5% ± 13.2% | 20.5% ± 12.9% | 24.7% ± 3.8% | 2.0 | 11.4 | 20% |
| dqn_mix | DQN + opponent mix (as PPO mix), 0.82M steps, buffer 200k | 3 | 87.4% ± 8.4% | 17.6% ± 3.9% | 28.9% ± 6.2% | 2.2 | 11.4 | 32% |
| v1_dqn_mix20k_best | DQN 20k, opponent mix 50/25/25 (best) | 1 | 94.2% | 30.0% | 35.4% | 1.8 | 12.0 | 16% |
| v1_legacy_dqn20k | DQN 20k episodes, self-play (best) | 1 | 90.4% | 29.6% | 38.0% | 2.4 | 12.6 | 55% |
| v1_legacy_dqn20k_double | Double DQN 20k (best) | 1 | 91.6% | 30.2% | 34.4% | 2.0 | 11.2 | 14% |
| v1_legacy_ppo1k | PPO 760 updates, self-play (best; run stopped early) | 1 | 93.2% | 31.6% | 26.6% | 1.9 | 14.9 | 33% |
