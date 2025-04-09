from jumanji.types import TimeStep
from functools import partial
from typing import List, Dict, Any, Tuple
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax as tfp
from utils import flatten_state
import wandb
from mLN.environment import DynamicSpectrumEnv


# Constants
NUM_BS = 3  
NUM_BANDS = 4  
NUM_USERS = 5  
NUM_POWER_LEVELS = 5  

# Hyperparameters
META_LR = 1e-3
INNER_LR = 0.1
META_BATCH_SIZE = 4
NUM_INNER_STEPS = 1
NUM_META_ITERS = 1000
ROLLOUT_LENGTH = 50
DISCOUNT_FACTOR = 0.99  
NUM_META_BATCHES = 10


tfd = tfp.distributions

def residual_block(x, hidden_dim):
    h = hk.Linear(hidden_dim)(x)
    h = jax.nn.relu(h)
    h = hk.Linear(hidden_dim)(h)
    return jax.nn.relu(x + h)

class MLPNetwork(hk.Module):
    def __init__(self, num_bs, num_bands, num_power_levels, hidden_dim=64, num_blocks=3):
        super().__init__()
        self.num_bs = num_bs
        self.num_bands = num_bands
        self.num_power_levels = num_power_levels
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

    def __call__(self, x):
        x = hk.Flatten()(x)
        x = hk.Linear(self.hidden_dim)(x)
        x = jax.nn.relu(x)
        for _ in range(self.num_blocks):
            x = residual_block(x, self.hidden_dim)
        logits = hk.Linear(self.num_bs * self.num_bands * self.num_power_levels)(x)
        logits = logits.reshape(-1, self.num_bs * self.num_bands, self.num_power_levels)
        return logits

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

def policy_fn(obs, num_bs, num_bands, num_power_levels):
    logits = MLPNetwork(num_bs, num_bands, num_power_levels)(obs)
    logits = jnp.clip(logits, -10.0, 10.0)
    return tfd.Categorical(logits=logits)

def value_fn(obs):
    return ValueNetwork()(obs)

def make_networks(num_bs, num_bands, num_power_levels):
    policy = hk.without_apply_rng(hk.transform(
        lambda obs: policy_fn(obs, num_bs, num_bands, num_power_levels)
    ))
    value = hk.without_apply_rng(hk.transform(value_fn))
    return policy, value

ObsNormalizerState = tuple  # (mean, var, count, eps, min_count)

def init_obs_normalizer(obs_dim: int) -> ObsNormalizerState:
    return (
        jnp.zeros(obs_dim),  # mean
        jnp.ones(obs_dim) * 10.0 ,   # var
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

def normalize_obs(state: ObsNormalizerState, obs: jnp.ndarray) -> jnp.ndarray:
    mean, var, count, eps, min_count = state
    if count < min_count:
        return obs
    normed = (obs - mean) / jnp.sqrt(var + eps)
    return jnp.clip(normed, -3.0, 3.0)

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

def normalize_reward(state: RewardNormalizerState, rewards: jnp.ndarray) -> jnp.ndarray:
    [mean, var, count, eps, min_count] = state
    if count < min_count:
        return rewards
    normed = (obs - mean) / jnp.sqrt(var + eps)
    normed = jnp.clip(normed, -3.0, 3.0)
    return normed

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
    state, timestep = task_env.reset(key)
    observations = []
    actions = []
    rewards = []
    values = []
    dones = []
    
    # Get initial observation
    obs = flatten_state(state)
    norm_obs = normalize_obs(obs_norm_state, obs)
    
    for _ in range(num_steps):
        key, action_key = jax.random.split(key)
        
        # Get action from policy
        action_dist = policy_apply(policy_params, norm_obs)
        action = action_dist.sample(seed=action_key)
        action_flat = action.reshape(-1)  # Ensure action has correct shape
        
        # Get value estimate
        value = value_apply(value_params, norm_obs)
        value = jnp.reshape(value, ())
        
        # Step environment
        next_state, next_timestep = task_env.step(state, action_flat)
        reward = next_timestep.reward
        norm_reward = normalize_reward(reward_norm_state, reward)
        
        # Store trajectory elements
        observations.append(norm_obs)
        actions.append(action)
        rewards.append(norm_reward)
        values.append(value)
        dones.append(next_timestep.last())
        
        # Get next observation
        next_obs = flatten_state(next_state)
        norm_obs = normalize_obs(obs_norm_state, next_obs)
        
        # Update state
        state, timestep = next_state, next_timestep
        if next_timestep.last():
            break
    
    # Handle empty trajectories
    if not observations:
        # Return empty trajectory with the right structure
        return {
            'observations': jnp.zeros((0, obs.shape[0])),
            'actions': jnp.zeros((0, action.shape[0])),
            'rewards': jnp.zeros(0),
            'values': jnp.zeros(0),
            'dones': jnp.zeros(0, dtype=bool),
            'final_value': jnp.array(0.0),
        }
    
    # Calculate final value for the last state
    final_value = value_apply(value_params, norm_obs).squeeze()
    
    # Convert lists to JAX arrays
    traj = {
        'observations': jnp.stack(observations),
        'actions': jnp.stack(actions),
        'rewards': jnp.array(rewards),
        'values': jnp.array(values),
        'dones': jnp.array(dones),
        'final_value': final_value,
    }
    return traj

def compute_gae(rewards, values, dones, final_value, gamma=0.99, lambda_=0.95):
    """
    Compute Generalized Advantage Estimation (GAE) for a trajectory.
    
    Args:
        rewards: JAX array of shape (T,), rewards at each timestep.
        values: JAX array of shape (T,), value estimates at each timestep.
        dones: JAX array of shape (T,), boolean flags (True if episode ends).
        final_value: JAX array (scalar), value of the last state.
        gamma: Discount factor.
        lambda_: GAE lambda parameter.
    
    Returns:
        returns: JAX array of shape (T,), discounted returns.
        advantages: JAX array of shape (T,), normalized advantages.
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
        update_value = td_error + gamma * lambda_ * next_advantage * mask
        update_value = jnp.nan_to_num(update_value, nan=0.0)  
        advantages = jnp.clip(advantages, -5.0, 5.0)
        advantages = advantages.at[t].set(update_value)
        returns = returns.at[t].set(rewards[t] + gamma * next_value * mask)
        
        next_value = values[t]
        next_advantage = advantages[t]
    
    # Normalize advantages
    adv_mean = jnp.mean(advantages)
    adv_std = jnp.std(advantages) + 1e-4
    advantages = (advantages - adv_mean) / adv_std
    advantages = jnp.nan_to_num(advantages, nan=0.0)
    
    return returns, advantages

def compute_inner_loss(policy_params, value_params, policy_apply, value_apply, traj):
    obs = traj['observations']
    actions = traj['actions']
    returns = traj['returns']
    advantages = traj['advantages']
    
    action_dist = policy_apply(policy_params, obs)
    entropy = action_dist.entropy()
    log_probs = action_dist.log_prob(actions)
    
    advantages = advantages[:, jnp.newaxis]
    policy_loss = -jnp.mean(log_probs * advantages) - 0.01 * jnp.mean(entropy)
    
    values = value_apply(value_params, obs)
    values = jnp.nan_to_num(values, nan=0.0)
    value_loss = jnp.mean(jnp.square(returns - values))
    
    combined_loss = policy_loss + 0.5 * value_loss
    
    return combined_loss, policy_loss, value_loss

def inner_adaptation(policy_params, value_params, policy_apply, value_apply, traj, inner_lr, inner_steps):
    prepared_traj = prepare_trajectory(traj, value_params, value_apply)
    
    def adaptation_step(step, curr_params):
        curr_policy_params, curr_value_params = curr_params
        grad_fn = jax.value_and_grad(lambda p: compute_inner_loss(
            p[0], p[1], policy_apply, value_apply, prepared_traj
        )[0], has_aux=False)
        
        inner_loss, grads = grad_fn((curr_policy_params, curr_value_params))
        policy_grads, value_grads = grads
        policy_grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), policy_grads)
        value_grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), value_grads)
        
        new_policy_params = jax.tree.map(
            lambda p, g: p - inner_lr * g, curr_policy_params, policy_grads
        )
        new_value_params = jax.tree.map(
            lambda p, g: p - inner_lr * g, curr_value_params, value_grads
        )
        
        return (new_policy_params, new_value_params)
    
    adapted_params = (policy_params, value_params)
    for i in range(inner_steps):
        adapted_params = adaptation_step(i, adapted_params)
    
    return adapted_params


def prepare_trajectory(traj, value_params, value_apply):
    returns, advantages = compute_gae(
        traj['rewards'], 
        traj['values'], 
        traj['dones'], 
        traj['final_value']
    )
    return {**traj, 'returns': returns, 'advantages': advantages}

def compute_meta_objective(init_params, train_traj, test_traj, 
                          policy_apply, value_apply, inner_lr, inner_steps):
    init_policy_params, init_value_params = init_params
    adapted_params = inner_adaptation(
        init_policy_params, init_value_params, 
        policy_apply, value_apply, 
        train_traj, inner_lr, inner_steps
    )
    adapted_policy_params, adapted_value_params = adapted_params
    
    test_traj_prepared = prepare_trajectory(test_traj, adapted_value_params, value_apply)
    test_loss, policy_loss, value_loss = compute_inner_loss(
        adapted_policy_params, adapted_value_params, 
        policy_apply, value_apply, test_traj_prepared
    )
    
    return test_loss, (policy_loss, value_loss)

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

def compute_meta_objective_for_task(
    params: Tuple[Any, Any],
    env: Any,
    policy_apply: Any,
    value_apply: Any,
    task_key: jnp.ndarray,
    inner_lr: float,
    inner_steps: int,
    obs_norm_state: ObsNormalizerState,
    reward_norm_state: RewardNormalizerState
) -> Tuple[jnp.ndarray, ObsNormalizerState, RewardNormalizerState]:
    task_key, train_key, test_key = jax.random.split(task_key, 3)
    policy_params, value_params = params
    
    # Sample task
    task_env = sample_task(env, task_key)
    
    # Sample training trajectories
    train_traj = sample_trajectories(
        task_env, policy_params, value_params, policy_apply, value_apply, train_key,
        obs_norm_state, reward_norm_state
    )
    
    # Prepare trajectory with returns and advantages
    train_traj_prepared = prepare_trajectory(train_traj, value_params, value_apply)
    
    # Inner adaptation
    adapted_params = inner_adaptation(
        policy_params, value_params, policy_apply, value_apply, train_traj_prepared, inner_lr, inner_steps
    )
    
    # Sample test trajectories with adapted params
    test_traj = sample_trajectories(
        task_env, adapted_params[0], adapted_params[1], policy_apply, value_apply, test_key,
        obs_norm_state, reward_norm_state
    )
    
    # Prepare test trajectory
    test_traj_prepared = prepare_trajectory(test_traj, adapted_params[1], value_apply)
    
    # Compute meta loss
    meta_loss, _, _ = compute_inner_loss(
        adapted_params[0], adapted_params[1], policy_apply, value_apply, test_traj_prepared
    )
    
    # Update normalizers with both train and test data
    all_obs = jnp.concatenate([train_traj['observations'], test_traj['observations']], axis=0)
    all_rewards = jnp.concatenate([train_traj['rewards'], test_traj['rewards']], axis=0)
    
    new_obs_norm_state = update_obs_normalizer(obs_norm_state, all_obs)
    new_reward_norm_state = update_reward_normalizer(reward_norm_state, all_rewards)
    
    return meta_loss, new_obs_norm_state, new_reward_norm_state


def adapt_to_task(params, env, policy_apply, value_apply, task_key, inner_lr, inner_steps, obs_norm_state, reward_norm_state):
    # Simple gradient descent adaptation (implement your actual inner loop)
    adapted_params = params
    for _ in range(inner_steps):
        loss_fn = lambda p: compute_meta_objective_for_task(
            p, env, policy_apply, value_apply, task_key, inner_lr, inner_steps,
            obs_norm_state, reward_norm_state
        )[0]
        loss, grads = jax.value_and_grad(loss_fn)(adapted_params)
        adapted_params = jax.tree.map(lambda p, g: p - inner_lr * g, adapted_params, grads)
    return adapted_params

def train_maml(
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
    dim: int,
    key: jnp.ndarray,
    eval_interval: int = 10,  
    num_eval_tasks: int = 5,
    wandb_project: str = "maml-training",
    wandb_name: str = None,
    use_wandb: bool = True
):
    params = (policy_params, value_params)
    obs_norm_state = init_obs_normalizer(dim)
    reward_norm_state = init_reward_normalizer()
    
    if use_wandb:
        # Initialize W&B
        config = {
            "num_tasks": num_tasks,
            "inner_lr": inner_lr,
            "inner_steps": inner_steps,
            "meta_lr": meta_lr,
            "num_iterations": num_iterations,
            "dim": dim,
            "eval_interval": eval_interval,
            "num_eval_tasks": num_eval_tasks
        }
        
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config=config
        )
    
    # Initialize optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip(1.0),
        optax.adam(meta_lr)
    )
    opt_state = optimizer.init(params)
    
    # Metrics storage
    meta_losses = []
    avg_grad_norms = []
    task_performances = []
    
    for iteration in range(num_iterations):
        key, subkey = jax.random.split(key)
        task_keys = jax.random.split(subkey, num_tasks)
        
        meta_loss_sum = 0.0
        all_grads = None
        valid_tasks = 0
        task_returns = []
        
        for i in range(num_tasks):
            task_key = jax.random.split(task_keys[i], 2)  
            train_key, test_key = task_key[0], task_key[1]
            
            meta_loss_fn = lambda p: compute_meta_objective_for_task(
                p, env, policy_apply, value_apply, task_key, inner_lr, inner_steps,
                obs_norm_state, reward_norm_state
            )[0]
            
            meta_loss, grads = jax.value_and_grad(meta_loss_fn)(params)
            if not jnp.isfinite(meta_loss):  # Skip invalid losses
                continue
            
            # Get train return before adaptation
            task_env = sample_task(env, train_key)
            pre_traj = sample_trajectories(
                task_env, params[0], params[1], policy_apply, value_apply, train_key,
                obs_norm_state, reward_norm_state
            )
            train_return = jnp.mean(pre_traj['rewards']) if len(pre_traj['rewards']) > 0 else 0.0
            
            # Get test return after adaptation
            adapted_params = adapt_to_task(
                params, env, policy_apply, value_apply, test_key,
                inner_lr, inner_steps, obs_norm_state, reward_norm_state
            )
            test_env = sample_task(env, test_key)
            post_traj = sample_trajectories(
                test_env, adapted_params[0], adapted_params[1], policy_apply, value_apply, test_key,
                obs_norm_state, reward_norm_state
            )
            test_return = jnp.mean(post_traj['rewards']) if len(post_traj['rewards']) > 0 else 0.0
            
            task_returns.append((float(train_return), float(test_return)))
            
            meta_loss_sum += meta_loss
            valid_tasks += 1
            if all_grads is None:
                all_grads = grads
            else:
                all_grads = jax.tree.map(lambda g1, g2: g1 + g2, all_grads, grads)
        
        if valid_tasks == 0:
            print(f"Iteration {iteration}: No valid tasks, skipping update")
            continue
        
        # Average gradients and losses
        avg_meta_loss = meta_loss_sum / valid_tasks
        avg_grads = jax.tree.map(lambda g: g / valid_tasks, all_grads)
        grad_norm = jnp.sqrt(jax.tree_util.tree_reduce(
            lambda acc, x: acc + jnp.sum(jnp.square(x)), avg_grads, 0.0
        ))
        
        # Apply updates
        updates, opt_state = optimizer.update(avg_grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        # Update normalizers
        if iteration % 5 == 0:
            norm_update_key = jax.random.fold_in(key, iteration)
            task_env = sample_task(env, norm_update_key)
            sample_traj = sample_trajectories(
                task_env, params[0], params[1], policy_apply, value_apply, norm_update_key,
                obs_norm_state, reward_norm_state
            )
            if len(sample_traj['observations']) > 0:
                obs_norm_state = update_obs_normalizer(obs_norm_state, sample_traj['observations'])
                reward_norm_state = update_reward_normalizer(reward_norm_state, sample_traj['rewards'])

        # Evaluate task performance periodically
        if iteration % eval_interval == 0:
            eval_key, subkey = jax.random.split(key)
            eval_task_keys = jax.random.split(subkey, num_eval_tasks)
            total_reward = 0.0
            valid_eval_tasks = 0
            
            for eval_key in eval_task_keys:
                # Perform inner loop adaptation
                adapted_params = adapt_to_task(
                    params, env, policy_apply, value_apply, eval_key,
                    inner_lr, inner_steps, obs_norm_state, reward_norm_state
                )
                # Evaluate adapted parameters
                eval_env = sample_task(env, eval_key)
                traj = sample_trajectories(
                    eval_env, adapted_params[0], adapted_params[1],
                    policy_apply, value_apply, eval_key,
                    obs_norm_state, reward_norm_state
                )
                if len(traj['rewards']) > 0:
                    total_reward += jnp.mean(traj['rewards'])
                    valid_eval_tasks += 1
            
            avg_performance = total_reward / valid_eval_tasks if valid_eval_tasks > 0 else 0.0
            task_performances.append(float(avg_performance))
            print(f"Iteration {iteration}, Task Performance: {avg_performance:.4f}")
        
       # Store metrics
        meta_losses.append(float(avg_meta_loss))
        avg_grad_norms.append(float(grad_norm))
        
        # Log metrics to W&B
        if use_wandb:
            metrics = {
                "meta_loss": avg_meta_loss,
                "grad_norm": grad_norm,
                "valid_tasks": valid_tasks
            }
            
            # Log individual task returns
            if task_returns and iteration % 10 == 0:
                for i, (train_ret, test_ret) in enumerate(task_returns[:5]):  # Log up to 5 tasks
                    metrics.update({
                        f"task_{i}_train_return": train_ret,
                        f"task_{i}_test_return": test_ret,
                        f"task_{i}_improvement": test_ret / (train_ret + 1e-8),
                    })
            
            wandb.log(metrics)    
            
        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Meta Loss: {avg_meta_loss:.4f}, Grad Norm: {grad_norm:.4f}")
    
    if use_wandb:
        wandb.run.summary["final_meta_loss"] = avg_meta_loss
        wandb.run.summary["final_grad_norm"] = grad_norm
        wandb.run.summary["final_task_performance"] = avg_performance
        wandb.run.summary["best_task_performance"] = max(task_performances)
        wandb.run.summary["best_meta_loss"] = min(meta_losses)
        wandb.run.summary["best_grad_norm"] = min(avg_grad_norms)
        
        wandb.finish()  # Close W&B
    
    return params, {
        'meta_losses': meta_losses,
        'avg_grad_norms': avg_grad_norms,
        'task_performances': task_performances
    }