import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax as tfp
from typing import Any, Tuple, Dict, List
from utils import flatten_state
import wandb
from mLN.environment import DynamicSpectrumEnv

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
            # Initialize hidden state
            batch_size = x.shape[0]
            hidden_state = lstm.initial_state(batch_size)
        
        # Process observation with LSTM
        x, new_hidden_state = lstm(x, hidden_state)
        
        # Remove batch dimension if needed
        if x.shape[0] == 1:
            x = x.squeeze(0)
        
        # Output logits for action distribution
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

def normalize_obs(state: ObsNormalizerState, obs: jnp.ndarray) -> jnp.ndarray:
    mean, var, count, eps, min_count = state
    if count < min_count:
        return obs
    normed = (obs - mean) / jnp.sqrt(var + eps)
    return jnp.clip(normed, -3.0, 3.0)

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

def normalize_reward(state: RewardNormalizerState, rewards: jnp.ndarray) -> jnp.ndarray:
    mean, var, count, eps, min_count = state
    if count < min_count:
        return rewards
    normed = (rewards - mean) / jnp.sqrt(var + eps)
    return jnp.clip(normed, -3.0, 3.0)

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
    log_probs = []
    
    # Get initial observation
    obs = flatten_state(state)
    norm_obs = normalize_obs(obs_norm_state, obs)
    
    # Initialize LSTM hidden state
    hidden_state = None
    hidden_states = []
    
    for _ in range(num_steps):
        key, action_key = jax.random.split(key)
        
        # Get action from recurrent policy
        action_dist, new_hidden_state = policy_apply(policy_params, norm_obs, hidden_state)
        action = action_dist.sample(seed=action_key)
        log_prob = action_dist.log_prob(action)
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
        log_probs.append(log_prob)
        hidden_states.append(hidden_state)
        
        # Update hidden state for next step
        hidden_state = new_hidden_state
        
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
            'log_probs': jnp.zeros(0),
            'hidden_states': [],
            'final_value': jnp.array(0.0),
            'final_hidden_state': None
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
        'log_probs': jnp.array(log_probs),
        'hidden_states': hidden_states,
        'final_value': final_value,
        'final_hidden_state': hidden_state
    }
    return traj

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
    
    # Forward pass policy network with stored hidden states
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
    
    # Ensure proper shape matching
    log_probs = jnp.array(log_probs)
    entropies = jnp.array(entropies)
    values = jnp.array(values)
    
    # For multi-dimensional log_probs, take the mean along the dimensions
    if len(log_probs.shape) > 1:
        log_probs = log_probs.mean(axis=tuple(range(1, len(log_probs.shape))))
    
    if len(old_log_probs.shape) > 1:
        old_log_probs = old_log_probs.mean(axis=tuple(range(1, len(old_log_probs.shape))))
    
    # Ensure entropies have the right shape
    if len(entropies.shape) > 1:
        entropies = entropies.mean(axis=tuple(range(1, len(entropies.shape))))
    
    # Compute policy loss with clipping
    ratio = jnp.exp(log_probs - old_log_probs)
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    
    # Ensure ratio and advantages have compatible shapes for broadcasting
    policy_loss1 = -ratio * advantages
    policy_loss2 = -clipped_ratio * advantages
    policy_loss = jnp.mean(jnp.maximum(policy_loss1, policy_loss2))
    
    # Value loss
    value_loss = jnp.mean(jnp.square(returns - values))
    
    # Entropy loss
    entropy_loss = -jnp.mean(entropies)
    
    # Combined loss
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
    
    def adaptation_step(step, curr_params):
        curr_policy_params, curr_value_params = curr_params
        
        def loss_fn(params):
            pol_params, val_params = params
            loss, _ = compute_ppo_loss(
                pol_params, val_params, policy_apply, value_apply, prepared_traj, clip_ratio
            )
            return loss
        
        grads = jax.grad(loss_fn, has_aux=False)((curr_policy_params, curr_value_params))
        policy_grads, value_grads = grads
        
        # Gradient clipping for stability
        policy_grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), policy_grads)
        value_grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), value_grads)
        
        new_policy_params = jax.tree_map(
            lambda p, g: p - inner_lr * g, curr_policy_params, policy_grads
        )
        new_value_params = jax.tree_map(
            lambda p, g: p - inner_lr * g, curr_value_params, value_grads
        )
        
        return (new_policy_params, new_value_params)
    
    adapted_params = (policy_params, value_params)
    for i in range(inner_steps):
        adapted_params = adaptation_step(i, adapted_params)
    
    return adapted_params

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
    wandb_project: str = "recurrent-maml-ppo",
    wandb_name: str = None,
    use_wandb: bool = True
):
    params = (policy_params, value_params)
    obs_norm_state = init_obs_normalizer(obs_dim)
    reward_norm_state = init_reward_normalizer()
    
    # Initialize optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip(1.0),
        optax.adam(meta_lr)
    )
    
    opt_state = optimizer.init(params)
    
    meta_losses = []
    policy_losses = []
    value_losses = []
    entropy_losses = []

    # Initialize W&B
    if use_wandb:
        config = {
            "num_tasks": num_tasks,
            "inner_lr": inner_lr,
            "inner_steps": inner_steps,
            "meta_lr": meta_lr,
            "num_iterations": num_iterations,
            "obs_dim": obs_dim,
            "clip_ratio": clip_ratio,
        }

        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config=config
        )
        
    valid_iteration_count = 0
    
    
    for iteration in range(num_iterations):
        key, subkey = jax.random.split(key)
        task_keys = jax.random.split(subkey, num_tasks)
        
        # Compute meta-objective and gradients for each task
        meta_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_loss_sum = 0.0
        all_grads = None
        successful_tasks = 0
        task_returns = []
        
        for i in range(num_tasks):
            task_key = task_keys[i]
            task_env = sample_task(env, task_key)
            
            # Split keys for sampling
            task_key, train_key, test_key = jax.random.split(task_keys[i], 3)
            
            # Sample training trajectories
            train_traj = sample_trajectories(
                task_env, params[0], params[1], policy_apply, value_apply, train_key,
                obs_norm_state, reward_norm_state
            )
            
            # Skip if we got an empty trajectory
            if len(train_traj['observations']) == 0:
                continue
                
            # Inner adaptation with PPO
            try:
                adapted_params = ppo_inner_adaptation(
                    params[0], params[1], policy_apply, value_apply,
                    train_traj, inner_lr, inner_steps, clip_ratio
                )
                
                # Sample test trajectories with adapted params
                test_traj = sample_trajectories(
                    task_env, adapted_params[0], adapted_params[1], policy_apply, value_apply, test_key,
                    obs_norm_state, reward_norm_state
                )
                
                # Skip if we got an empty test trajectory
                if len(test_traj['observations']) == 0:
                    continue

                # Log task-specific metrics
                train_return = jnp.sum(train_traj['rewards'])
                test_return = jnp.sum(test_traj['rewards'])
                task_returns.append((train_return, test_return))
                
                # Meta-gradient computation
                def meta_objective_fn(p):
                    return compute_meta_objective(
                        p, train_traj, test_traj, policy_apply, value_apply,
                        inner_lr, inner_steps, clip_ratio
                    )
                
                (meta_loss, (p_loss, v_loss, e_loss)), grads = jax.value_and_grad(
                    meta_objective_fn, has_aux=True
                )(params)
                
                meta_loss_sum += meta_loss
                policy_loss_sum += p_loss
                value_loss_sum += v_loss
                entropy_loss_sum += e_loss
                
                # Accumulate gradients
                if all_grads is None:
                    all_grads = grads
                else:
                    all_grads = jax.tree.map(lambda g1, g2: g1 + g2, all_grads, grads)
                
                # Update normalizers with both train and test data
                all_obs = jnp.concatenate([train_traj['observations'], test_traj['observations']], axis=0)
                all_rewards = jnp.concatenate([train_traj['rewards'], test_traj['rewards']], axis=0)
                
                obs_norm_state = update_obs_normalizer(obs_norm_state, all_obs)
                reward_norm_state = update_reward_normalizer(reward_norm_state, all_rewards)
                successful_tasks += 1
                valid_iteration_count += 1
                
            except Exception as e:
                print(f"Error in task {i}: {e}")
                continue
        
        # Skip update if no valid gradients were computed
        if all_grads is None:
            print(f"Iteration {iteration}: No valid gradients, skipping update")
            continue
        
        # Average gradients and losses
        num_valid_tasks = max(1, successful_tasks )  # Avoid division by zero
        avg_meta_loss = meta_loss_sum / num_valid_tasks
        avg_policy_loss = policy_loss_sum / num_valid_tasks
        avg_value_loss = value_loss_sum / num_valid_tasks
        avg_entropy_loss = entropy_loss_sum / num_valid_tasks
        avg_grads = jax.tree.map(lambda g: g / num_valid_tasks, all_grads)
        
        # Apply updates
        updates, opt_state = optimizer.update(avg_grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        # Store losses
        meta_losses.append(avg_meta_loss)
        policy_losses.append(avg_policy_loss)
        value_losses.append(avg_value_loss)
        entropy_losses.append(avg_entropy_loss)

        task_success_rate = successful_tasks / num_tasks
        
        # Calculate stats for task returns
        if task_returns:
            train_returns = [tr for tr, _ in task_returns]
            test_returns = [tr for _, tr in task_returns]
            mean_train_return = jnp.mean(jnp.array(train_returns))
            mean_test_return = jnp.mean(jnp.array(test_returns))
            
            # Adaptation improvement (test/train ratio)
            adaptation_improvement = mean_test_return / (mean_train_return + 1e-8)  # Avoid division by zero
        else:
            mean_train_return = 0.0
            mean_test_return = 0.0
            adaptation_improvement = 0.0
        
        # Log metrics to W&B
        if use_wandb:
            wandb.log({
                "iteration": iteration,
                "meta_loss": avg_meta_loss,
                "policy_loss": avg_policy_loss,
                "value_loss": avg_value_loss,
                "entropy_loss": avg_entropy_loss,
                "task_success_rate": task_success_rate,
                "mean_train_return": mean_train_return,
                "mean_test_return": mean_test_return,
                "adaptation_improvement": adaptation_improvement,
                "num_successful_tasks": successful_tasks,
                "gradient_norm": optax.global_norm(avg_grads),
                "param_norm": optax.global_norm(params),
            }, step=iteration)
            
            # Log individual task returns
            if task_returns and iteration % 10 == 0:  
                for i, (train_ret, test_ret) in enumerate(task_returns[:5]):  
                    wandb.log({
                        f"task_{i}_train_return": train_ret,
                        f"task_{i}_test_return": test_ret,
                        f"task_{i}_improvement": test_ret / (train_ret + 1e-8),
                    }, step=iteration)

        if iteration % 5 == 0:
            print(f"Iteration {iteration}, Meta Loss: {avg_meta_loss:.4f}, "
                  f"Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}, "
                  f"Success Rate: {task_success_rate:.2f}, Test Return: {mean_test_return:.2f}")
    
    # Finish W&B run
    if use_wandb:
        # Create summary metrics
        wandb.run.summary["final_meta_loss"] = meta_losses[-1]
        wandb.run.summary["best_meta_loss"] = min(meta_losses)
        wandb.run.summary["final_task_success_rate"] = task_success_rate
        wandb.run.summary["total_valid_iterations"] = valid_iteration_count
        
        # Finish the run
        wandb.finish()
    return params, {
        'meta_losses': meta_losses,
        'policy_losses': policy_losses,
        'value_losses': value_losses,
        'entropy_losses': entropy_losses
    }

# def train_recurrent_maml_ppo(
#     env: Any,
#     policy_params: Any,
#     value_params: Any,
#     policy_apply: Any,
#     value_apply: Any,
#     num_tasks: int,
#     inner_lr: float,
#     inner_steps: int,
#     meta_lr: float,
#     num_iterations: int,
#     obs_dim: int,
#     key: jnp.ndarray,
#     clip_ratio: float = 0.2
# ):
#     params = (policy_params, value_params)
#     obs_norm_state = init_obs_normalizer(obs_dim)
#     reward_norm_state = init_reward_normalizer()
    
#     # Initialize optimizer with gradient clipping
#     optimizer = optax.chain(
#         optax.clip(1.0),
#         optax.adam(meta_lr)
#     )
    
#     opt_state = optimizer.init(params)
    
#     meta_losses = []
#     policy_losses = []
#     value_losses = []
#     entropy_losses = []
    
#     for iteration in range(num_iterations):
#         key, subkey = jax.random.split(key)
#         task_keys = jax.random.split(subkey, num_tasks)
        
#         # Compute meta-objective and gradients for each task
#         meta_loss_sum = 0.0
#         policy_loss_sum = 0.0
#         value_loss_sum = 0.0
#         entropy_loss_sum = 0.0
#         all_grads = None
        
#         for i in range(num_tasks):
#             task_key = task_keys[i]
#             task_env = sample_task(env, task_key)
            
#             # Split keys for sampling
#             task_key, train_key, test_key = jax.random.split(task_key, 3)
            
#             # Sample training trajectories
#             train_traj = sample_trajectories(
#                 task_env, params[0], params[1], policy_apply, value_apply, train_key,
#                 obs_norm_state, reward_norm_state
#             )
            
#             # Skip if we got an empty trajectory
#             if len(train_traj['observations']) == 0:
#                 continue
                
#             # Inner adaptation with PPO
#             try:
#                 adapted_params = ppo_inner_adaptation(
#                     params[0], params[1], policy_apply, value_apply,
#                     train_traj, inner_lr, inner_steps, clip_ratio
#                 )
                
#                 # Sample test trajectories with adapted params
#                 test_traj = sample_trajectories(
#                     task_env, adapted_params[0], adapted_params[1], policy_apply, value_apply, test_key,
#                     obs_norm_state, reward_norm_state
#                 )
                
#                 # Skip if we got an empty test trajectory
#                 if len(test_traj['observations']) == 0:
#                     continue
                
#                 # Meta-gradient computation
#                 def meta_objective_fn(p):
#                     return compute_meta_objective(
#                         p, train_traj, test_traj, policy_apply, value_apply,
#                         inner_lr, inner_steps, clip_ratio
#                     )
                
#                 (meta_loss, (p_loss, v_loss, e_loss)), grads = jax.value_and_grad(
#                     meta_objective_fn, has_aux=True
#                 )(params)
                
#                 meta_loss_sum += meta_loss
#                 policy_loss_sum += p_loss
#                 value_loss_sum += v_loss
#                 entropy_loss_sum += e_loss
                
#                 # Accumulate gradients
#                 if all_grads is None:
#                     all_grads = grads
#                 else:
#                     all_grads = jax.tree_map(lambda g1, g2: g1 + g2, all_grads, grads)
                
#                 # Update normalizers with both train and test data
#                 all_obs = jnp.concatenate([train_traj['observations'], test_traj['observations']], axis=0)
#                 all_rewards = jnp.concatenate([train_traj['rewards'], test_traj['rewards']], axis=0)
                
#                 obs_norm_state = update_obs_normalizer(obs_norm_state, all_obs)
#                 reward_norm_state = update_reward_normalizer(reward_norm_state, all_rewards)
            
#             except Exception as e:
#                 print(f"Error in task {i}: {e}")
#                 continue
        
#         # Skip update if no valid gradients were computed
#         if all_grads is None:
#             print(f"Iteration {iteration}: No valid gradients, skipping update")
#             continue
        
#         # Average gradients and losses
#         num_valid_tasks = max(1, i + 1)  # Avoid division by zero
#         avg_meta_loss = meta_loss_sum / num_valid_tasks
#         avg_policy_loss = policy_loss_sum / num_valid_tasks
#         avg_value_loss = value_loss_sum / num_valid_tasks
#         avg_entropy_loss = entropy_loss_sum / num_valid_tasks
#         avg_grads = jax.tree_map(lambda g: g / num_valid_tasks, all_grads)
        
#         # Apply updates
#         updates, opt_state = optimizer.update(avg_grads, opt_state)
#         params = optax.apply_updates(params, updates)
        
#         # Store losses
#         meta_losses.append(avg_meta_loss)
#         policy_losses.append(avg_policy_loss)
#         value_losses.append(avg_value_loss)
#         entropy_losses.append(avg_entropy_loss)
        
#         if iteration % 5 == 0:
#             print(f"Iteration {iteration}, Meta Loss: {avg_meta_loss:.4f}, "
#                   f"Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}")
    
#     return params, {
#         'meta_losses': meta_losses,
#         'policy_losses': policy_losses,
#         'value_losses': value_losses,
#         'entropy_losses': entropy_losses
#     }


