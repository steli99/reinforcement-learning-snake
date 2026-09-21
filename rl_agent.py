"""Synchronous n-step advantage actor-critic for the supplied Snake task."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import tensorflow as tf


def set_seed(seed):
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def build_actor_critic(observation_shape):
    inputs = tf.keras.Input(shape=observation_shape, name='observation')
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    logits = tf.keras.layers.Dense(4, kernel_initializer=tf.keras.initializers.Orthogonal(0.01),
                                   name='action_logits')(x)
    value = tf.keras.layers.Dense(1, name='state_value')(x)
    model = tf.keras.Model(inputs, [logits, value], name='snake_actor_critic')
    # Both views share variables; only the combined model is optimized.
    agent = tf.keras.Model(inputs, logits, name='actor_logits')
    critic = tf.keras.Model(inputs, value, name='critic_value')
    return model, agent, critic


def discounted_returns(rewards, dones, bootstrap, gamma):
    """n-step targets; dones marks a reset, not a mere wall hit or self-bite."""
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=bool)
    returns = np.empty_like(rewards)
    running = np.asarray(bootstrap, dtype=np.float32).copy()
    for t in range(len(rewards)-1, -1, -1):
        running = rewards[t] + gamma * (~dones[t]) * running
        returns[t] = running
    return returns


class A2C:
    def __init__(self, observation_shape, learning_rate=7e-4, gamma=.9,
                 entropy_weight=.01, value_weight=.5):
        self.model, self.agent, self.value = build_actor_critic(observation_shape)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.gamma, self.entropy_weight, self.value_weight = gamma, entropy_weight, value_weight

    @tf.function(reduce_retracing=True)
    def sample(self, observations):
        logits, values = self.model(observations, training=False)
        actions = tf.squeeze(tf.random.categorical(logits, 1, dtype=tf.int32), axis=1)
        return actions, tf.squeeze(values, axis=1)

    @tf.function(reduce_retracing=True)
    def update(self, observations, actions, targets):
        with tf.GradientTape() as tape:
            logits, values = self.model(observations, training=True)
            values = tf.squeeze(values, axis=1)
            advantage = tf.stop_gradient(targets-values)
            # Normalize only the policy advantage, not the critic target.
            policy_advantage = (advantage-tf.reduce_mean(advantage))/(tf.math.reduce_std(advantage)+1e-8)
            negative_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits)
            policy_loss = tf.reduce_mean(negative_log_prob * policy_advantage)
            value_loss = .5*tf.reduce_mean(tf.square(tf.stop_gradient(targets)-values))
            probs = tf.nn.softmax(logits)
            entropy = -tf.reduce_mean(tf.reduce_sum(probs*tf.nn.log_softmax(logits), axis=1))
            loss = policy_loss+self.value_weight*value_loss-self.entropy_weight*entropy
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, norm = tf.clip_by_global_norm(gradients, .5)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, policy_loss, value_loss, entropy, norm

    def train(self, env, updates=1000, rollout_steps=16, log_every=100):
        state = env.to_state()
        history = []
        for update in range(1, updates+1):
            observations, actions, rewards, dones = [], [], [], []
            events = {'fruit': 0, 'wall': 0, 'self_bite': 0, 'win': 0}
            for _ in range(rollout_steps):
                action, _ = self.sample(tf.convert_to_tensor(state))
                action = action.numpy()
                new_state, reward, done, info = env.step(action)
                observations.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                for key in events:
                    events[key] += int(info[key].sum())
                state = new_state
            _, bootstrap = self.model(tf.convert_to_tensor(state), training=False)
            targets = discounted_returns(rewards, dones, bootstrap.numpy().ravel(), self.gamma)
            losses = self.update(tf.convert_to_tensor(np.concatenate(observations)),
                                 tf.convert_to_tensor(np.concatenate(actions), dtype=tf.int32),
                                 tf.convert_to_tensor(targets.ravel(), dtype=tf.float32))
            transitions = rollout_steps*env.n_boards
            row = {'update': update, 'transitions': update*transitions,
                   'mean_reward': float(np.mean(rewards)), 'mean_length': float(info['length'].mean())}
            for key, value in zip(['loss','policy_loss','value_loss','entropy','gradient_norm'], losses):
                row[key] = float(value.numpy())
            row.update({key+'_rate': count/transitions for key, count in events.items()})
            history.append(row)
            if update == 1 or update % log_every == 0:
                mean_reward = np.mean([h['mean_reward'] for h in history[-log_every:]])
                print(f'Update {update:4d}/{updates}: reward/step={mean_reward:.4f}, '
                      f'entropy={row["entropy"]:.3f}, length={row["mean_length"]:.2f}', flush=True)
        return history


def evaluate_policy(env_factory, model=None, seeds=(100,101,102), n_boards=64,
                    steps=1000, greedy=False):
    """Evaluate raw, undiscounted reward per move; never update model weights."""
    results = []
    for seed in seeds:
        env = env_factory(n_boards, seed)
        action_rng = np.random.default_rng(seed+10000)
        totals = dict(reward=0., fruit=0, wall=0, self_bite=0, win=0, length=0.)
        state = env.to_state()
        for _ in range(steps):
            if model is None:
                actions = action_rng.integers(0,4,n_boards)
            else:
                logits, _ = model(tf.convert_to_tensor(state), training=False)
                if greedy:
                    actions = np.argmax(logits.numpy(), axis=1)
                else:
                    probs = tf.nn.softmax(logits).numpy()
                    draws = action_rng.random(n_boards)
                    actions = (draws[:,None] > np.cumsum(probs,axis=1)).sum(axis=1)
                    actions = np.minimum(actions,3)
            state, rewards, _, info = env.step(actions)
            totals['reward'] += float(rewards.sum())
            for key in ['fruit','wall','self_bite','win']:
                totals[key] += int(info[key].sum())
            totals['length'] += float(info['length'].sum())
        count = steps*n_boards
        results.append({'seed':seed,'transitions':count,'mean_reward':totals['reward']/count,
                        'fruit_rate':totals['fruit']/count,'wall_rate':totals['wall']/count,
                        'self_bite_rate':totals['self_bite']/count,'win_rate':totals['win']/count,
                        'mean_length':totals['length']/count})
    return results
