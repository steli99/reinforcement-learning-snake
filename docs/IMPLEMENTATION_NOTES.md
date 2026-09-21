# Implementation notes

## Completed algorithm

The template left the actor, value network, action selection, loss, gradients, and optimizer parameters undefined. The completed project uses synchronous n-step advantage actor–critic. The policy and value function share a two-layer network, with independent output heads. A separate action-value Q network is unnecessary for this algorithm and is intentionally omitted.

Returns bootstrap from the final rollout observation. They stop across filled-board resets. Actor advantages and target returns are detached; the critic receives gradients through its prediction. Actor advantages are standardized across the rollout batch. The entropy bonus encourages exploration, and gradient clipping limits the global gradient norm.

The optimized objective is discounted reward with gamma 0.9. Evaluation instead reports undiscounted reward per move over a fixed horizon. These are related but distinct quantities.

## Environment fixes and interface

- Removed private `keras.api._v2.keras` imports and legacy optimizer usage. Observation encoding uses NumPy; training uses the public TensorFlow/Keras API.
- Both `(N,)` and `(N,1)` action arrays are supported and validated. Caller-owned action arrays are not modified.
- Fixed blocked moves: the original code could insert the stationary head into the body and drop the tail, leaving an inconsistent body list. A wall hit now leaves both board and body unchanged.
- Added explicit win/reset flags to prevent bootstrapping a return from the next episode into the preceding one.
- Preserved the original body-truncation behavior on a self-bite. A collision is not a terminal death, and moving onto a tail cell is still a self-bite under these rules.
- Preserved all five reward constants, playable-board geometry, four-channel observation encoding, and random fruit placement on empty cells.
- Full and partial environments share the same transition implementation. The local view pads outside-board areas with wall values.
- Per-environment random generators make resets and evaluation independent of unrelated global NumPy sampling.
- `move(actions)` returns a NumPy float32 reward column of shape `(N,1)`; the old environment returned a TensorFlow tensor. TensorFlow accepts the NumPy output directly. `step(actions)` additionally returns the next observation, flat rewards, done flags, and event information.
- `step` automatically resets a won board before returning its observation. `done=True` identifies this boundary. Only wins are terminal in this assignment; evaluation horizon truncation does not change the dynamics.

## Scope and limitations

The file called “fully observable” exposes the occupancy grid, not the ordered internal body list. When the snake touches itself, occupancy need not uniquely identify tail order. It should not be described as a proven Markov observation. The local observation has additional ambiguity because distant fruit is invisible. The feedforward agents are observation-based baselines, not recurrent or belief-state solutions.

The training budget is stated explicitly and differs from the unfinished template. One training seed is used for each observation variant. Each fixed policy is evaluated on three fresh seeds with the same board counts and horizons. Standard deviation across evaluation seeds measures evaluation variability only, not robustness to retraining. No held-out evaluation is used to choose a checkpoint: the final update is evaluated.

Greedy inference may exhibit loops even when stochastic inference earns reward. Reward improvement does not establish reliable board completion: the event table reports fruit, wall, self-bite, and win rates separately. A self-bite can shorten a snake and make later fruit easier to reach; that possibility comes from the supplied reward and transition rules.
