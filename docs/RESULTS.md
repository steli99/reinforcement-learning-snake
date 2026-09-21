# Verified Snake results

The complete notebook was executed sequentially from its first code cell to its last. Both agents trained for 1,000 updates × 16 rollout steps × 64 boards = **1,024,000 transitions each**. No training or evaluation cell was skipped.

Notebook execution time: 122.3 seconds in the recorded CPU environment. This is environment-specific, not a benchmark. The included runner executes Python cells and embeds their outputs/figures without launching a Jupyter server.

## Held-out policy evaluation

Each policy uses 192,000 evaluation moves (three fresh seeds, 64 boards, 1,000 moves). Rates are counts per move. Reward SD is across the three evaluation-seed means, not across independent training runs.

| Observation | Policy | Reward/move | Seed SD | Fruit/move | Wall/move | Self-bite/move | Wins/move | Mean length |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| full | A2C greedy | 0.12194 | 0.00334 | 0.27972 | 0.01886 | 0.08015 | 0.000000 | 3.92 |
| full | A2C stochastic | 0.12432 | 0.00028 | 0.28262 | 0.00140 | 0.08427 | 0.000000 | 3.75 |
| full | random | -0.01447 | 0.00012 | 0.01730 | 0.19954 | 0.01583 | 0.000000 | 1.07 |
| partial | A2C greedy | 0.11782 | 0.00030 | 0.25827 | 0.00000 | 0.05657 | 0.000000 | 6.11 |
| partial | A2C stochastic | 0.11788 | 0.00018 | 0.25841 | 0.00015 | 0.05657 | 0.000000 | 6.02 |
| partial | random | -0.01447 | 0.00012 | 0.01730 | 0.19954 | 0.01583 | 0.000000 | 1.07 |

## Interpretation

- Full observation: the stochastic learned policy changes reward per move by **+0.13879** relative to random.
- Partial observation: the stochastic learned policy changes reward per move by **+0.13235** relative to random.

No completed-board wins were observed for either trained policy in these evaluation runs. The demonstrated behavior is improved reward and fruit collection, not reliable board completion. Fruit collection, length, self-bites, and wins must be interpreted separately. The task rewards repeated fruit collection and makes self-bites nonterminal, so a high-return policy need not maximize snake length or complete boards. Greedy policies can lose the exploration provided by sampling and become stuck in loops. These measurements characterize one trained agent per observation type; three evaluation seeds do not demonstrate robustness to training initialization.

## Tests and artifacts

All 10 environment and learning tests passed. The tests cover fixed-seed initial states, fruit growth, unchanged bodies on wall hits, self-bite truncation, terminal wins and reset observations, agreement of full/partial dynamics, invalid-action handling, body invariants under random play, reset-aware return targets, and a finite weight update. See `tests.txt`.

The `.keras` files contain inference-ready model parameters. They do not include the external A2C optimizer state. Exact package versions are recorded in `environment.json`. Training traces and per-seed evaluation data are provided as CSVs; figures and a 120-frame sampled rollout are included.
