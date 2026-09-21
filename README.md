# Snake Reinforcement Learning

A completed TensorFlow project that trains a Snake agent with **n-step advantage actor–critic (A2C)** and evaluates full-board and partially observable policies against uniform random play.

![Trained policy rollout](artifacts/snake_policy.gif)

## Included

- [Executed notebook](main.ipynb): algorithm explanation, training, evaluation, and plots.
- [Actor–critic implementation](rl_agent.py): on-policy rollouts, detached return targets, entropy bonus, and gradient clipping.
- [Full-board environment](environments_fully_observable.py) and [local-view environment](environments_partially_observable.py).
- Trained `.keras` models, CSV training logs, evaluation results, and a policy GIF in `artifacts/`.
- [Verification and measured results](docs/RESULTS.md), [implementation notes](docs/IMPLEMENTATION_NOTES.md), and numerical/environment tests.

## The task

The supplied coursework uses a nonstandard Snake environment:

| Event | Reward | Behavior |
|---|---:|---|
| Ordinary move | 0 | Move forward |
| Eat fruit | +0.5 | Grow and place another fruit |
| Hit wall | −0.1 | Remain in place; continue playing |
| Bite own body | −0.2 | Truncate the body; continue playing |
| Fill the board | +1 | Reset that board and mark the episode boundary |

These dynamics are retained, with the wall-hit body corruption fixed. There is no action masking or reward shaping.

## Setup

Use Python 3.12 and a virtual environment. From this project folder:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` on Windows PowerShell, or `source .venv/bin/activate` on macOS/Linux.

```bash
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open `main.ipynb`. Its saved outputs can be inspected without retraining. To reproduce them, restart the kernel and run all cells from top to bottom. The notebook and its two environment modules must remain in the same project directory.

The checkpoint files use standard Keras layers and can be loaded with:

```python
import tensorflow as tf
from environments_fully_observable import OriginalSnakeEnvironment

env = OriginalSnakeEnvironment(1, 7, seed=123)
model = tf.keras.models.load_model('artifacts/actor_critic_full.keras', compile=False)
logits, value = model(tf.constant(env.to_state()))
action = tf.argmax(logits, axis=1).numpy()
observation, reward, done, info = env.step(action)
```

For the local policy, use `environments_partially_observable.OriginalSnakeEnvironment(1, 7, 2, seed=123)` and `actor_critic_partial.keras`. The notebook reports stochastic and greedy evaluation separately.

## Training and evaluation

| Setting | Value |
|---|---:|
| Board size, including border | 7 × 7 |
| Playable area | 5 × 5 |
| Local observation | 5 × 5 centered on the head |
| Parallel boards | 64 |
| Rollout length | 16 |
| Optimizer updates per agent | 1,000 |
| Training transitions per agent | 1,024,000 |
| Discount factor | 0.9 |
| Adam learning rate | 0.0007 |
| Hidden layers | 128, 128 (ReLU) |
| Entropy coefficient | 0.01 |
| Critic-loss coefficient | 0.5 |
| Gradient-norm limit | 0.5 |
| Training seed | 0 for each observation variant |
| Evaluation seeds | 100, 101, 102 |
| Evaluation per policy and observation variant | 192,000 moves |

The supplied unfinished notebook's 1,000-board × 5,000-step configuration has been replaced with this explicitly documented rollout-based budget. Training can take several minutes or longer on a CPU. Checkpoints are provided to avoid retraining just to inspect the learned policy.

The full-grid observation omits ordered tail history, and the local crop hides distant fruit. Both policies are memoryless; they do not solve all hidden-state ambiguity. Evaluation variation across three seeds is not variation across independent training runs. See the notebook for these limitations and the measured behavior.

## Verification

```bash
python -m unittest discover -s tests -v
```

The tests cover collision semantics, body invariants, fruit growth, win resets, partial observations, random-seed reproducibility, terminal return masking, and a finite parameter update.

For a complete execution without a Jupyter server:

```bash
python scripts/execute_notebook.py main.ipynb
```

This script updates notebook outputs, plots, checkpoints, and result files.

## GitHub

Create an empty repository named `reinforcement-learning-snake`, then upload this folder's **contents**, including the `artifacts/` folder. Do not upload just the ZIP. Alternatively:

```bash
git init -b main
git add .
git commit -m "Add trained Snake actor-critic project"
git remote add origin https://github.com/YOUR_USERNAME/reinforcement-learning-snake.git
git push -u origin main
```

## Attribution

Adapted from the three supplied coursework files. The implementation, fixes, documentation, and experiments were prepared with AI assistance. No ownership or redistribution license is asserted over the original course material.
