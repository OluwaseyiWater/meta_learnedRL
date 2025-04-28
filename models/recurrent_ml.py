%%writefile models/recurrent_ml.py
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax as tfp
from typing import Any, Tuple, Dict, List
from utils import flatten_state
import wandb
from mLN.environment import DynamicSpectrumEnv
from jax import lax
from functools import partial

tfd = tfp.distributions

def residual_block(x, hidden_dim):
    h = hk.Linear(hidden_dim)(x)
    h = jax.nn.relu(h)
    h = hk.Linear(hidden_dim)(h)
    return jax.nn.relu(x + h)

class RecurrentPolicyNetwork(hk.Module):
    def __init__(self, num_bs, num_bands, num_power_levels, hidden_dim=64, lstm_hidden_dim=32):
        super().__init__()
        self.num_bs = num_bs
        self.num_bands = num_bands
        self.num_power_levels = num_power_levels
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim

    def __call__(self, obs, hidden_state):
        # Ensure obs is flattened
        x = hk.Flatten()(obs)

        # Project observation to hidden_dim
        x = hk.Linear(self.hidden_dim)(x)
        x = jax.nn.relu(x)

        # Add batch dimension if needed
        if len(x.shape) == 1:
            x = x[None, :]  # (hidden_dim,) -> (1, hidden_dim)

        # LSTM layer
        lstm = hk.LSTM(self.lstm_hidden_dim)
        if hidden_state is None:
            batch_size = x.shape[0]
            hidden_state = lstm.initial_state(batch_size)
        else:
            # Restore batch dimension to hidden state if missing
            if hidden_state.hidden.ndim == 1:
                hidden_state = hk.LSTMState(
                    hidden=hidden_state.hidden[None, :],
                    cell=hidden_state.cell[None, :]
                )

        # Process observation with LSTM
        x, new_hidden_state = lstm(x, hidden_state)

        # After LSTM, now safe to squeeze batch dim
        if new_hidden_state.hidden.ndim == 2:
            new_hidden_state = hk.LSTMState(
                hidden=new_hidden_state.hidden.squeeze(0),
                cell=new_hidden_state.cell.squeeze(0)
            )

        # Remove batch dimension if needed
        if x.shape[0] == 1:
            x = x.squeeze(0)

        # Output logits
        logits = hk.Linear(self.num_bs * self.num_bands * self.num_power_levels)(x)
        logits = logits.reshape(-1, self.num_bs * self.num_bands, self.num_power_levels)

        return logits, new_hidden_state


class ValueNetwork(hk.Module):
    def __init__(self, hidden_dim=64, num_blocks=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

    def __call__(self, x):
        x = hk.Flatten()(x)
        x = hk.Linear(self.hidden_dim)(x)
        x = jax.nn.relu(x)
        for _ in range(self.num_blocks):
            x = residual_block(x, self.hidden_dim)
        return hk.Linear(1)(x)

def recurrent_policy_fn(obs, hidden_state, num_bs, num_bands, num_power_levels):
    logits, new_state = RecurrentPolicyNetwork(num_bs, num_bands, num_power_levels)(obs, hidden_state)
    logits = jnp.clip(logits, -10.0, 10.0)
    return tfd.Categorical(logits=logits), new_state

def value_fn(obs):
    return ValueNetwork()(obs)

def make_networks(num_bs, num_bands, num_power_levels):
    policy = hk.without_apply_rng(hk.transform(
        lambda obs, hidden_state: recurrent_policy_fn(obs, hidden_state, num_bs, num_bands, num_power_levels)
    ))
    value = hk.without_apply_rng(hk.transform(value_fn))
    return policy, value


# Observation Normalizer
ObsNormalizerState = tuple  # (mean, var, count, eps, min_count)

def init_obs_normalizer(obs_dim: int) -> ObsNormalizerState:
    return (
        jnp.zeros(obs_dim),  # mean
        jnp.ones(obs_dim) * 10.0,  # var
        jnp.array(0),        # count
        jnp.array(1e-4),     # eps
        jnp.array(100)       # min_count
    )

def update_obs_normalizer(state: ObsNormalizerState, obs_batch: jnp.ndarray) -> ObsNormalizerState:
    mean, var, count, eps, min_count = state
    batch_mean = jnp.mean(obs_batch, axis=0)
    batch_var = jnp.var(obs_batch, axis=0)
    batch_count = obs_batch.shape[0]
    
    delta = batch_mean - mean
    total_count = count + batch_count
    
    new_mean = mean + delta * batch_count / total_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * count * batch_count / total_count
    new_var = M2 / total_count
    
    return (new_mean, new_var, total_count, eps, min_count)


def normalize_obs(state, obs):
    mean, var, count, eps, min_count = state
    normed = (obs - mean) / jnp.sqrt(var + eps)
    normed = jnp.clip(normed, -3.0, 3.0)
    do_norm = count >= min_count
    # broadcasts scalar do_norm across obs’s shape
    return jnp.where(do_norm, normed, obs)

# Reward Normalizer
RewardNormalizerState = tuple  # (mean, var, count, eps, min_count)

def init_reward_normalizer() -> RewardNormalizerState:
    return (
        jnp.array(0.0),  # mean
        jnp.array(10.0),  # var
        jnp.array(0),    # count
        jnp.array(1e-4), # eps
        jnp.array(100)   # min_count
    )

def update_reward_normalizer(state: RewardNormalizerState, rewards: jnp.ndarray) -> RewardNormalizerState:
    mean, var, count, eps, min_count = state
    batch_mean = jnp.mean(rewards)
    batch_var = jnp.var(rewards)
    batch_count = rewards.shape[0]
    
    delta = batch_mean - mean
    total_count = count + batch_count
    
    new_mean = mean + delta * batch_count / total_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * count * batch_count / total_count
    new_var = M2 / total_count
    
    return (new_mean, new_var, total_count, eps, min_count)

def normalize_reward(state, rewards):
    mean, var, count, eps, min_count = state
    normed = (rewards - mean) / jnp.sqrt(var + eps)
    normed = jnp.clip(normed, -3.0, 3.0)
    do_norm = count >= min_count
    return jnp.where(do_norm, normed, rewards)

def sample_trajectories(
    task_env: Any,
    policy_params: Any,
    value_params: Any,
    policy_apply: Any,
    value_apply: Any,
    key: jnp.ndarray,
    obs_norm_state: ObsNormalizerState,
    reward_norm_state: RewardNormalizerState,
    num_steps: int = 10
) -> dict:
    def safe_squeeze(x):
        return jnp.squeeze(x, axis=0) if x.ndim == 2 and x.shape[0] == 1 else x

    state, timestep = task_env.reset(key)
    obs = flatten_state(state)
    norm_obs = normalize_obs(obs_norm_state, obs)

    _, hidden_state = policy_apply(policy_params, norm_obs, None)

    def step_fn(carry, _):
        state, hidden_state, norm_obs, key = carry

        key, action_key = jax.random.split(key)
        dist, new_hidden_state = policy_apply(policy_params, norm_obs, hidden_state)
        action = dist.sample(seed=action_key)
        log_prob = dist.log_prob(action)
        value = value_apply(value_params, norm_obs).squeeze()

        action_flat = action.reshape(-1)
        next_state, next_timestep = task_env.step(state, action_flat)

        # normalized reward
        norm_reward = normalize_reward(reward_norm_state, next_timestep.reward)

        # compute violations
        sinr = task_env._compute_sinr(next_state)
        sv = jnp.sum(sinr < task_env.min_sinr)
        lv = jnp.sum(next_state.qos_metrics[:, 0] > task_env.max_latency)

        next_obs = flatten_state(next_state)
        next_norm_obs = normalize_obs(obs_norm_state, next_obs)
        done = next_timestep.last()

        new_carry = (next_state, new_hidden_state, next_norm_obs, key)
        outputs = (
            norm_obs,
            action,
            norm_reward,
            value,
            done,
            log_prob,
            safe_squeeze(hidden_state.hidden),
            safe_squeeze(hidden_state.cell),
            sv,
            lv
        )
        return new_carry, outputs

    carry = (state, hidden_state, norm_obs, key)
    carry, outputs = jax.lax.scan(step_fn, carry, xs=None, length=num_steps)

    (observations, actions, rewards, values, dones, log_probs,
     hiddens, cells, sinr_violations, qos_violations) = outputs

    # truncate on done
    if dones.any():
        idx = int(jnp.argmax(dones)) + 1
        observations     = observations[:idx]
        actions          = actions[:idx]
        rewards          = rewards[:idx]
        values           = values[:idx]
        dones            = dones[:idx]
        log_probs        = log_probs[:idx]
        hiddens          = hiddens[:idx]
        cells            = cells[:idx]
        sinr_violations  = sinr_violations[:idx]
        qos_violations   = qos_violations[:idx]

    hidden_states = [hk.LSTMState(hidden=h, cell=c) for h, c in zip(hiddens, cells)]
    final_value   = value_apply(value_params, carry[2]).squeeze()

    return {
        'observations':      observations,
        'actions':           actions,
        'rewards':           rewards,
        'values':            values,
        'dones':             dones,
        'log_probs':         log_probs,
        'hidden_states':     hidden_states,
        'final_value':       final_value,
        'final_hidden_state': carry[1],
        'sinr_violations':   sinr_violations,
        'qos_violations':    qos_violations,
    }


def compute_gae(rewards, values, dones, final_value, gamma=0.99, lambda_=0.95):
    """
    Compute Generalized Advantage Estimation (GAE) for a trajectory.
    """
    episode_length = len(rewards)
    
    # Handle empty trajectories
    if episode_length == 0:
        return jnp.array([]), jnp.array([])
    
    advantages = jnp.zeros_like(rewards)
    returns = jnp.zeros_like(rewards)
    
    next_value = final_value
    next_advantage = 0.0
    
    for t in reversed(range(episode_length)):
        mask = 1.0 - dones[t].astype(jnp.float32)  # Explicit cast for safety
        td_error = rewards[t] + gamma * next_value * mask - values[t]
        advantages = advantages.at[t].set(td_error + gamma * lambda_ * next_advantage * mask)
        returns = returns.at[t].set(advantages[t] + values[t])
        
        next_value = values[t]
        next_advantage = advantages[t]
    
    # Normalize advantages
    adv_mean = jnp.mean(advantages)
    adv_std = jnp.std(advantages) + 1e-4
    advantages = (advantages - adv_mean) / adv_std
    
    return returns, advantages

def prepare_trajectory(traj, value_params, value_apply):
    returns, advantages = compute_gae(
        traj['rewards'], 
        traj['values'], 
        traj['dones'], 
        traj['final_value']
    )
    return {**traj, 'returns': returns, 'advantages': advantages}

def compute_ppo_loss(
    policy_params,
    value_params,
    policy_apply,
    value_apply,
    traj,
    clip_ratio=0.2,
    value_coeff=0.5,
    entropy_coeff=0.01
):
    obs = traj['observations']
    actions = traj['actions']
    returns = traj['returns']
    advantages = traj['advantages']
    old_log_probs = traj['log_probs']
    hidden_states = traj['hidden_states']

    values = []
    log_probs = []
    entropies = []

    for i in range(len(obs)):
        action_dist, _ = policy_apply(policy_params, obs[i], hidden_states[i])
        log_prob = action_dist.log_prob(actions[i])
        entropy = action_dist.entropy()
        value = value_apply(value_params, obs[i])

        log_probs.append(log_prob)
        entropies.append(entropy)
        values.append(value.squeeze())

    log_probs = jnp.array(log_probs)
    entropies = jnp.array(entropies)
    values = jnp.array(values)

    if len(log_probs.shape) > 1:
        log_probs = log_probs.mean(axis=tuple(range(1, len(log_probs.shape))))

    if len(old_log_probs.shape) > 1:
        old_log_probs = old_log_probs.mean(axis=tuple(range(1, len(old_log_probs.shape))))

    if len(entropies.shape) > 1:
        entropies = entropies.mean(axis=tuple(range(1, len(entropies.shape))))

    ratio = jnp.exp(log_probs - old_log_probs)
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)

    policy_loss1 = -ratio * advantages
    policy_loss2 = -clipped_ratio * advantages
    policy_loss = jnp.mean(jnp.maximum(policy_loss1, policy_loss2))

    value_loss = jnp.mean(jnp.square(returns - values))
    entropy_loss = -jnp.mean(entropies)

    total_loss = policy_loss + value_coeff * value_loss + entropy_coeff * entropy_loss

    return total_loss, (policy_loss, value_loss, entropy_loss)



def ppo_inner_adaptation(
    policy_params,
    value_params,
    policy_apply,
    value_apply,
    traj,
    inner_lr,
    inner_steps,
    clip_ratio=0.2
):
    prepared_traj = prepare_trajectory(traj, value_params, value_apply)

    def adaptation_step(i, curr_params):
        curr_policy_params, curr_value_params = curr_params

        def loss_fn(params):
            pol_params, val_params = params
            loss, _ = compute_ppo_loss(
                pol_params, val_params, policy_apply, value_apply, prepared_traj, clip_ratio
            )
            return loss

        grads = jax.grad(loss_fn, has_aux=False)(curr_params)
        policy_grads, value_grads = grads

        # Clip gradients
        policy_grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), policy_grads)
        value_grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), value_grads)

        new_policy_params = jax.tree_map(lambda p, g: p - inner_lr * g, curr_policy_params, policy_grads)
        new_value_params = jax.tree_map(lambda p, g: p - inner_lr * g, curr_value_params, value_grads)

        return (new_policy_params, new_value_params)

    adapted_params = (policy_params, value_params)
    adapted_params = lax.fori_loop(0, inner_steps, adaptation_step, adapted_params)

    return adapted_params

@partial(jax.jit, static_argnums=(3,4,5,6,7))
def compute_meta_objective(
    init_params,
    train_traj,
    test_traj,
    policy_apply,
    value_apply,
    inner_lr,
    inner_steps,
    clip_ratio=0.2
):
    init_policy_params, init_value_params = init_params

    # Inner adaptation with PPO
    adapted_params = ppo_inner_adaptation(
        init_policy_params, init_value_params,
        policy_apply, value_apply,
        train_traj, inner_lr, inner_steps, clip_ratio
    )

    adapted_policy_params, adapted_value_params = adapted_params

    # Prepare test trajectory
    test_traj_prepared = prepare_trajectory(test_traj, adapted_value_params, value_apply)

    # Compute test loss with adapted parameters
    test_loss, (policy_loss, value_loss, entropy_loss) = compute_ppo_loss(
        adapted_policy_params, adapted_value_params,
        policy_apply, value_apply, test_traj_prepared, clip_ratio
    )

    return test_loss, (policy_loss, value_loss, entropy_loss)

batched_meta_objective = jax.vmap(
    compute_meta_objective,
    in_axes=(None, 0, 0, None, None, None, None, None)  # Only varying train_traj, test_traj
)

def sample_task(env, key):
    variation = jax.random.uniform(key, (), minval=0.8, maxval=1.2)
    new_max_interference = env.max_interference * variation
    new_env = DynamicSpectrumEnv(
        num_bs=env.num_bs,
        num_users=env.num_users,
        num_bands=env.num_bands,
        max_steps=env.max_steps,
        max_latency=env.max_latency,
        max_power=env.max_power,
        num_power_levels=env.num_power_levels,
        power_levels=env.power_levels,
        fading_coherence=env.fading_coherence,
        max_interference=new_max_interference,
        min_sinr=env.min_sinr
    )
    return new_env

# Helper to stack trajectories

def stack_trajectories(trajs):
    keys = trajs[0].keys()
    stacked = {}
    for k in keys:
        if isinstance(trajs[0][k], jnp.ndarray):
            stacked[k] = [traj[k] for traj in trajs]
        else:
            stacked[k] = [traj[k] for traj in trajs]
    return stacked

# Patched train_recurrent_maml_ppo with manual looping

def train_recurrent_maml_ppo(
    env: Any,
    policy_params: Any,
    value_params: Any,
    policy_apply: Any,
    value_apply: Any,
    num_tasks: int,
    inner_lr: float,
    inner_steps: int,
    meta_lr: float,
    num_iterations: int,
    obs_dim: int,
    key: jnp.ndarray,
    clip_ratio: float = 0.2,
    eval_interval: int = 10,
    num_eval_tasks: int = 5,
    wandb_project: str = "recurrent-maml-ppo",
    wandb_name: str = None,
    use_wandb: bool = True
) -> Tuple[Tuple[Any,Any], Dict[str, list]]:
    params = (policy_params, value_params)
    obs_norm_state = init_obs_normalizer(obs_dim)
    reward_norm_state = init_reward_normalizer()

    optimizer = optax.chain(
        optax.clip(1.0),
        optax.adam(meta_lr)
    )
    opt_state = optimizer.init(params)

    # History metrics
    meta_losses = []
    avg_pre_rewards = []
    avg_post_rewards = []
    avg_reward_improvements = []
    avg_pre_sinrs = []
    avg_post_sinrs = []
    avg_sinr_improvements = []
    avg_pre_qoss = []
    avg_post_qoss = []
    avg_qos_improvements = []

    if use_wandb:
        config = {
            "num_tasks": num_tasks,
            "inner_lr": inner_lr,
            "inner_steps": inner_steps,
            "meta_lr": meta_lr,
            "num_iterations": num_iterations,
            "obs_dim": obs_dim,
            "clip_ratio": clip_ratio,
            "eval_interval": eval_interval,
            "num_eval_tasks": num_eval_tasks
        }
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config=config
        )

    # Pre-generate keys for all iterations and tasks
    total_task_keys = jax.random.split(key, num_iterations * num_tasks)

    # Vectorized update (no vmap over ragged trajectories)
    for iteration in range(num_iterations):
        # slice keys
        slice_keys = total_task_keys[iteration * num_tasks : (iteration + 1) * num_tasks]

        all_grads = None
        meta_loss_sum = 0.0
        task_returns = []
        successful_tasks = 0

        # inner loop over tasks
        for tk in slice_keys:
            task_env = sample_task(env, tk)
            train_traj = sample_trajectories(
                task_env, params[0], params[1], policy_apply, value_apply, tk,
                obs_norm_state, reward_norm_state
            )
            if len(train_traj['observations']) == 0:
                continue
            # pre-adaptation metrics
            pre_reward = float(jnp.mean(train_traj['rewards']))
            pre_sinr = float(jnp.mean(train_traj.get('sinr_violations', 0)))
            pre_qos  = float(jnp.mean(train_traj.get('qos_violations', 0)))

            adapted_params = ppo_inner_adaptation(
                params[0], params[1], policy_apply, value_apply,
                train_traj, inner_lr, inner_steps, clip_ratio
            )

            # post-adaptation trajectories
            key_inner = jax.random.split(tk)[1]
            post_traj = sample_trajectories(
                task_env, adapted_params[0], adapted_params[1], policy_apply, value_apply,
                key_inner, obs_norm_state, reward_norm_state
            )
            if len(post_traj['observations']) == 0:
                continue
            post_reward = float(jnp.mean(post_traj['rewards']))
            post_sinr = float(jnp.mean(post_traj.get('sinr_violations', 0)))
            post_qos  = float(jnp.mean(post_traj.get('qos_violations', 0)))

            task_returns.append((pre_reward, post_reward))
            # accumulate improvements
            avg_pre_rewards.append(pre_reward)
            avg_post_rewards.append(post_reward)
            avg_reward_improvements.append(post_reward - pre_reward)
            avg_pre_sinrs.append(pre_sinr)
            avg_post_sinrs.append(post_sinr)
            avg_sinr_improvements.append(pre_sinr - post_sinr)
            avg_pre_qoss.append(pre_qos)
            avg_post_qoss.append(post_qos)
            avg_qos_improvements.append(pre_qos - post_qos)

            # meta-loss and grad
            meta_loss, grads = jax.value_and_grad(
                lambda p: compute_meta_objective(
                    p, train_traj, post_traj, policy_apply, value_apply,
                    inner_lr, inner_steps, clip_ratio
                )[0]
            )(params)
            meta_loss_sum += meta_loss
            if all_grads is None:
                all_grads = grads
            else:
                all_grads = jax.tree_map(lambda a, b: a + b, all_grads, grads)

            successful_tasks += 1

        if successful_tasks == 0:
            print(f"Iteration {iteration}: no valid tasks, skip update")
            continue

        avg_meta_loss = meta_loss_sum / successful_tasks
        meta_losses.append(avg_meta_loss)
        avg_grads = jax.tree_map(lambda g: g / successful_tasks, all_grads)

        updates, opt_state = optimizer.update(avg_grads, opt_state)
        params = optax.apply_updates(params, updates)

        # evaluation logging every eval_interval
        if iteration % eval_interval == 0:
            # average metrics over last num_eval_tasks pre/post
            idx0 = -num_eval_tasks
            stats = {
                "avg_pre_reward": sum(avg_pre_rewards[idx0:]) / num_eval_tasks,
                "avg_post_reward": sum(avg_post_rewards[idx0:]) / num_eval_tasks,
                "avg_reward_improvement": sum(avg_reward_improvements[idx0:]) / num_eval_tasks,
                "avg_pre_sinr_violation": sum(avg_pre_sinrs[idx0:]) / num_eval_tasks,
                "avg_post_sinr_violation": sum(avg_post_sinrs[idx0:]) / num_eval_tasks,
                "avg_sinr_improvement": sum(avg_sinr_improvements[idx0:]) / num_eval_tasks,
                "avg_pre_qos_violation": sum(avg_pre_qoss[idx0:]) / num_eval_tasks,
                "avg_post_qos_violation": sum(avg_post_qoss[idx0:]) / num_eval_tasks,
                "avg_qos_improvement": sum(avg_qos_improvements[idx0:]) / num_eval_tasks,
            }
            if use_wandb:
                wandb.log({"iteration": iteration, "meta_loss": avg_meta_loss, **stats}, step=iteration)
            print(f"[Iter {iteration}] meta_loss={avg_meta_loss:.3f} pre_r={stats['avg_pre_reward']:.3f} post_r={stats['avg_post_reward']:.3f}")

    if use_wandb:
        wandb.finish()
    return params, {
        "meta_losses": meta_losses,
        "avg_pre_rewards": avg_pre_rewards,
        "avg_post_rewards": avg_post_rewards,
        "avg_reward_improvements": avg_reward_improvements,
        "avg_pre_sinrs": avg_pre_sinrs,
        "avg_post_sinrs": avg_post_sinrs,
        "avg_sinr_improvements": avg_sinr_improvements,
        "avg_pre_qoss": avg_pre_qoss,
        "avg_post_qoss": avg_post_qoss,
        "avg_qos_improvements": avg_qos_improvements,
    }




