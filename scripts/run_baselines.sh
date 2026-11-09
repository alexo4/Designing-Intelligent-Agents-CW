#!/usr/bin/env bash
# Train DQN and PPO on several seeds, then plot the learning curves.
#
#   bash scripts/run_baselines.sh            # seeds 0 1 2
#   SEEDS="0 1 2 3 4" bash scripts/run_baselines.sh
#
# Runs one job per algorithm at a time, in parallel across algorithms.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-0 1 2}"
PPO_UPDATES="${PPO_UPDATES:-300}"
DQN_EPISODES="${DQN_EPISODES:-3000}"
mkdir -p runs results

for algo in ppo dqn; do
  (
    for seed in $SEEDS; do
      out="runs/${algo}_s${seed}"
      if [[ -f "$out/${algo}.pt" ]]; then
        echo "skipping $out (already trained)"
        continue
      fi
      if [[ $algo == ppo ]]; then
        python -u train.py ppo --updates "$PPO_UPDATES" --seed "$seed" --out-dir "$out" > "$out.log" 2>&1
      else
        python -u train.py dqn --episodes "$DQN_EPISODES" --seed "$seed" --out-dir "$out" > "$out.log" 2>&1
      fi
      echo "finished $out"
    done
  ) &
done
wait

python plot.py runs/*/ --out results/learning_curves.png | tee results/summary.md
