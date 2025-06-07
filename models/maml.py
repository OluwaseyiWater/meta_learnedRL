from jumanji.types import TimeStep 
from functools import partial
from typing import Any, Callable, Tuple, Dict, List
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax as tfp
from utils import flatten_state 
import wandb
from mLN.environment import DynamicSpectrumEnv 
from jax import lax
import chex

# ==============================================================================
# CONSTANTS
# ==============================================================================
ROLLOUT_LENGTH = 50 
BANDWIDTH_HZ = 10e6
NOISE_FIGURE_DB = 7.0

tfd = tfp.distributions

# ==============================================================================
# NEURAL NETWORK ARCHITECTURES
# ==============================================================================

def residual_block(x, hidden_dim):
    """Residual block for deeper networks"""
    h = hk.Linear(hidden_dim)(x)
    h = jax.nn.relu(h)
    h = hk.Linear(hidden_dim)(h)
    return jax.nn.relu(x + h)


class MLPNetwork(hk.Module):
    """Policy network with residual blocks"""
    def __init__(self, num_bs, num_bands, num_power_levels, hidden_dim=64, num_blocks=2):
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
        # More conservative initialization like recurrent model
        logits = hk.Linear(
            self.num_bs * self.num_bands * self.num_power_levels,
            w_init=hk.initializers.TruncatedNormal(0.01),
            b_init=hk.initializers.Constant(0.0)
        )(x)
        logits = logits.reshape(-1, self.num_bs * self.num_bands, self.num_power_levels)
        return logits


class ValueNetwork(hk.Module):
    """Value network with residual blocks"""
    def __init__(self, hidden_dim=64, num_blocks=2):
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
    """Create policy distribution from observation"""
    logits = MLPNetwork(num_bs, num_bands, num_power_levels)(obs)
    # More conservative clipping like recurrent model
    logits = jnp.clip(logits, -3.0, 3.0)
    return tfd.Categorical(logits=logits)


def value_fn(obs):
    """Compute value from observation"""
    return ValueNetwork()(obs)


def make_networks(num_bs, num_bands, num_power_levels):
    """Create policy and value networks"""
    policy = hk.without_apply_rng(hk.transform(
        lambda obs: policy_fn(obs, num_bs, num_bands, num_power_levels)
    ))
    value = hk.without_apply_rng(hk.transform(value_fn))
    return policy, value

# ==============================================================================
# OBSERVATION NORMALIZATION
# ==============================================================================

ObsNormalizerState = tuple

def init_obs_normalizer(obs_dim: int) -> ObsNormalizerState:
    """Initialize observation normalizer state"""
    return (
        jnp.zeros(obs_dim),      # mean
        jnp.ones(obs_dim),       # variance
        jnp.array(0),            # count
        jnp.array(1e-4),         # epsilon
        jnp.array(50)            # min_count before normalization
    )


def update_obs_normalizer(state: ObsNormalizerState, obs_batch: jnp.ndarray) -> ObsNormalizerState:
    """Update normalizer statistics with new observations"""
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
    """Normalize observations using running statistics"""
    mean, var, count, eps, min_count = state
    normed = (obs - mean) / jnp.sqrt(var + eps + 1e-6)
    normed = jnp.clip(normed, -3.0, 3.0)
    do_norm = count >= min_count
    return jnp.where(do_norm, normed, jnp.clip(obs, -3.0, 3.0))

# ==============================================================================
# REWARD NORMALIZATION
# ==============================================================================

RewardNormalizerState = tuple

def init_reward_normalizer() -> RewardNormalizerState:
    """Initialize reward normalizer state"""
    return (
        jnp.array(-300.0),       # Initial mean
        jnp.array(100.0),        # Initial variance
        jnp.array(0),            # count
        jnp.array(1e-4),         # epsilon
        jnp.array(200)           # min_count before normalization
    )


def update_reward_normalizer(state: RewardNormalizerState, rewards: jnp.ndarray) -> RewardNormalizerState:
    """Update normalizer statistics with new rewards"""
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
    """Normalize rewards using running statistics"""
    mean, var, count, eps, min_count = state
    
    if count < min_count:
        return rewards / 150.0  
    
    normalized = (rewards - mean) / jnp.sqrt(var + eps + 1e-6)
    return jnp.clip(normalized, -50.0, 50.0)

# ==============================================================================
# TRAJECTORY SAMPLING
# ==============================================================================

def sample_trajectories_simple(
    task_env: Any, policy_params: Any, value_params: Any, policy_apply: Callable,
    value_apply: Callable, key: jnp.ndarray, obs_norm_state: Any,
    reward_norm_state: Any, num_steps: int = ROLLOUT_LENGTH
) -> Dict[str, jnp.ndarray]:
    """Simple non-JAX trajectory sampling like in recurrent model"""
    
    # Initialize
    state, timestep = task_env.reset(key)
    initial_obs = flatten_state(state)
    
    
    if not jnp.all(jnp.isfinite(initial_obs)):
        print("Warning: Invalid initial observation")
        return _create_empty_trajectory()
    
    initial_norm_obs = normalize_obs(obs_norm_state, initial_obs)
    
    # Storage
    observations = []
    actions = []
    rewards = []
    raw_rewards = []
    values = []
    dones = []
    sinr_violations = []
    qos_violations = []
    
    current_obs = initial_norm_obs
    current_state = state
    
    for step in range(num_steps):
        key, step_key = jax.random.split(key)
        
        # Get action
        action_dist = policy_apply(policy_params, current_obs)
        action = action_dist.sample(seed=step_key)
        
        # Get value
        value = value_apply(value_params, current_obs).squeeze()
        
        # Environment step
        action_flat = action.reshape(-1)
        next_state, next_timestep = task_env.step(current_state, action_flat)
        raw_reward = next_timestep.reward
        
       
        if not jnp.isfinite(raw_reward):
            raw_reward = jnp.array(-0.1)
            
        # Normalize reward
        norm_reward = normalize_reward(reward_norm_state, raw_reward)
        
       
        next_obs = flatten_state(next_state)
        if not jnp.all(jnp.isfinite(next_obs)):
            print("Warning: Invalid observation encountered")
            break
            
        next_norm_obs = normalize_obs(obs_norm_state, next_obs)
        
        # Calculate violations
        try:
            sinr_vals = task_env._compute_sinr(next_state)
            user_best_sinr = jnp.max(sinr_vals, axis=1)
            sinr_violation_count = jnp.sum(user_best_sinr < task_env.min_sinr)
            qos_violation_count = jnp.sum(next_state.qos_metrics[:, 0] > task_env.max_latency)
        except:
            sinr_violation_count = jnp.array(0.0)
            qos_violation_count = jnp.array(0.0)
        
        # Store data
        observations.append(current_obs)
        actions.append(action)
        rewards.append(norm_reward)
        raw_rewards.append(raw_reward)
        values.append(value)
        dones.append(next_timestep.last())
        sinr_violations.append(sinr_violation_count)
        qos_violations.append(qos_violation_count)
        
        # Update for next step
        current_obs = next_norm_obs
        current_state = next_state
        
        if next_timestep.last():
            break
    
    if len(observations) == 0:
        return _create_empty_trajectory()
    
    
    final_value = value_apply(value_params, current_obs).squeeze()
    
    return {
        'observations': jnp.array(observations),
        'actions': jnp.array(actions),
        'rewards': jnp.array(rewards),
        'raw_rewards': jnp.array(raw_rewards),
        'values': jnp.array(values),
        'dones': jnp.array(dones),
        'sinr_violations': jnp.array(sinr_violations),
        'qos_violations': jnp.array(qos_violations),
        'final_value': final_value
    }


def _create_empty_trajectory():
    """Helper function to create empty trajectory for error cases"""
    return {
        'observations': jnp.array([]).reshape(0, 70),  
        'actions': jnp.array([]).reshape(0, 15),  
        'rewards': jnp.array([]),
        'raw_rewards': jnp.array([]),
        'values': jnp.array([]),
        'dones': jnp.array([], dtype=bool),
        'sinr_violations': jnp.array([]),
        'qos_violations': jnp.array([]),
        'final_value': jnp.array(0.0)
    }

# ==============================================================================
# ADVANTAGE ESTIMATION
# ==============================================================================

def compute_gae(rewards, values, dones, final_value, gamma=0.99, lambda_=0.95):
    """Compute Generalized Advantage Estimation"""
    episode_length_fixed = rewards.shape[0]
    if episode_length_fixed == 0:
        return jnp.array([]), jnp.array([])
        
    advantages = jnp.zeros_like(rewards)
    next_value = final_value
    next_advantage = 0.0

    for t in reversed(range(episode_length_fixed)):
        is_last_step_of_episode = dones[t]
        mask = 1.0 - is_last_step_of_episode.astype(jnp.float32)
        td_error = rewards[t] + gamma * next_value * mask - values[t]
        advantages = advantages.at[t].set(td_error + gamma * lambda_ * next_advantage * mask)
        next_value = values[t]
        next_advantage = advantages[t]
    returns = advantages + values

    # Normalize advantages
    actual_seq_len = episode_length_fixed 
    indices = jnp.arange(actual_seq_len)
    first_true_idx_or_len = jnp.min(jnp.where(dones, indices, actual_seq_len))
    num_actual_steps = jnp.minimum(first_true_idx_or_len + 1, actual_seq_len)

    def normalize_advantages_fn(adv_in_operand):
        valid_mask = jnp.arange(actual_seq_len) < num_actual_steps
        valid_adv = jnp.where(valid_mask, adv_in_operand, 0.0)
        
        valid_count = jnp.sum(valid_mask)
        mean = jnp.sum(valid_adv) / jnp.maximum(valid_count, 1.0)
        
        squared_diff = jnp.square(valid_adv - mean) * valid_mask
        variance = jnp.sum(squared_diff) / jnp.maximum(valid_count, 1.0)
        std = jnp.sqrt(variance + 1e-8)
        
        norm_adv = (adv_in_operand - mean) / (std + 1e-8)
        new_adv = jnp.where(valid_mask, norm_adv, 0.0)
        return new_adv

    def zeros_advantages_fn(adv_in_operand):
        return jnp.zeros_like(adv_in_operand)

    advantages = lax.cond(
        num_actual_steps > 0, 
        normalize_advantages_fn,
        zeros_advantages_fn,
        advantages 
    )
    
    advantages = jnp.nan_to_num(advantages, nan=0.0, posinf=3.0, neginf=-3.0)
    returns = jnp.nan_to_num(returns, nan=0.0, posinf=10.0, neginf=-10.0)
    return returns, advantages


def prepare_trajectory(traj, value_params, value_apply):
    """Add returns and advantages to trajectory"""
    if traj['observations'].shape[0] == 0:
        return {**traj, 'returns': jnp.array([]), 'advantages': jnp.array([])}
    returns, advantages = compute_gae(
        traj['rewards'], traj['values'], traj['dones'], traj['final_value']
    )
    return {**traj, 'returns': returns, 'advantages': advantages}

# ==============================================================================
# LOSS COMPUTATION
# ==============================================================================

def compute_inner_loss(policy_params, value_params, policy_apply, value_apply, traj):
    """Compute policy and value loss for inner loop"""
    obs = traj['observations']
    actions = traj['actions']
    returns = traj['returns']
    advantages = traj['advantages']
    dones = traj['dones']

    if obs.shape[0] == 0:
        zero_loss = jnp.array(0.0, dtype=jnp.float32)
        return zero_loss, (zero_loss, zero_loss)

    # Get action distribution and values
    action_dist = policy_apply(policy_params, obs)
    entropy_terms = action_dist.entropy()
    log_prob_terms = action_dist.log_prob(actions)

    # Compute valid steps mask
    actual_seq_len = obs.shape[0]
    indices = jnp.arange(actual_seq_len)
    first_true_idx_or_len = jnp.min(jnp.where(dones, indices, actual_seq_len))
    num_valid_steps = jnp.minimum(first_true_idx_or_len + 1, actual_seq_len)
    valid_step_mask_float = (jnp.arange(actual_seq_len) < num_valid_steps).astype(jnp.float32)
    count_valid_steps = jnp.sum(valid_step_mask_float)
    safe_count_valid_steps = jnp.maximum(count_valid_steps, 1.0)

    # Handle dimension mismatch
    if log_prob_terms.ndim > 1 and advantages.ndim == 1:
        advantages_reshaped = advantages.reshape((-1,) + (1,) * (log_prob_terms.ndim - 1))
    else:
        advantages_reshaped = advantages
    
    mask_for_terms = valid_step_mask_float
    if log_prob_terms.ndim > 1:
        mask_for_terms = jnp.expand_dims(valid_step_mask_float, axis=tuple(range(1, log_prob_terms.ndim)))

    # Policy loss with entropy bonus
    policy_loss_val = -jnp.sum(log_prob_terms * advantages_reshaped * mask_for_terms) / safe_count_valid_steps \
                      - 0.01 * jnp.sum(entropy_terms * mask_for_terms) / safe_count_valid_steps
    policy_loss_val = jnp.clip(policy_loss_val, -10.0, 10.0)
    policy_loss_val = jnp.where(count_valid_steps > 0, policy_loss_val, 0.0)

    # Value loss
    values_pred = value_apply(value_params, obs).squeeze()
    values_pred = jnp.nan_to_num(values_pred, nan=0.0)
    value_loss_val = jnp.sum(jnp.square(returns - values_pred) * valid_step_mask_float) / safe_count_valid_steps
    value_loss_val = jnp.clip(value_loss_val, 0.0, 10.0)
    value_loss_val = jnp.where(count_valid_steps > 0, value_loss_val, 0.0)

    # Combined loss
    combined_loss = policy_loss_val + 0.3 * value_loss_val
    combined_loss = jnp.where(jnp.isfinite(combined_loss), combined_loss, 0.0)
    
    return combined_loss, (policy_loss_val, value_loss_val)

# ==============================================================================
# INNER LOOP ADAPTATION
# ==============================================================================

def inner_adaptation(policy_params, value_params, policy_apply, value_apply, traj, inner_lr, inner_steps):
    """Perform inner loop adaptation on a trajectory"""
    if traj['observations'].shape[0] == 0:
        return policy_params, value_params
        
    def adaptation_step(carry, _):
        curr_policy_params, curr_value_params = carry
        
        # Compute gradients
        grad_fn = jax.value_and_grad(
            lambda p_tuple: compute_inner_loss(
                p_tuple[0], p_tuple[1], policy_apply, value_apply, traj
            )[0], argnums=0, has_aux=False
        )
        inner_loss, grads_tuple = grad_fn((curr_policy_params, curr_value_params))
        policy_grads, value_grads = grads_tuple
        
        # Gradient clipping
        max_grad_norm = 0.5
        
        policy_grad_norm = optax.global_norm(policy_grads)
        value_grad_norm = optax.global_norm(value_grads)
        
        policy_clip_factor = jnp.minimum(1.0, max_grad_norm / (policy_grad_norm + 1e-8))
        value_clip_factor = jnp.minimum(1.0, max_grad_norm / (value_grad_norm + 1e-8))
        
        policy_grads = jax.tree_map(lambda g: g * policy_clip_factor, policy_grads)
        value_grads = jax.tree_map(lambda g: g * value_clip_factor, value_grads)
        
        
        effective_lr = inner_lr * 0.5
        new_policy_params = jax.tree_map(lambda p, g: p - effective_lr * g, curr_policy_params, policy_grads)
        new_value_params = jax.tree_map(lambda p, g: p - effective_lr * g, curr_value_params, value_grads)
        
        return (new_policy_params, new_value_params), inner_loss

    init_params_tuple = (policy_params, value_params)
    (adapted_policy_params, adapted_value_params), _ = jax.lax.scan(
        adaptation_step, init_params_tuple, None, length=inner_steps
    )
    return adapted_policy_params, adapted_value_params

# ==============================================================================
# META-LEARNING OBJECTIVES
# ==============================================================================

def compute_meta_objective(init_params, train_traj, test_traj,
                          policy_apply, value_apply, inner_lr, inner_steps):
    """Compute meta-objective: performance on test trajectory after adaptation"""
    init_policy_params, init_value_params = init_params
    
    # Adapt on training trajectory
    adapted_policy_params, adapted_value_params = inner_adaptation(
        init_policy_params, init_value_params,
        policy_apply, value_apply,
        train_traj, inner_lr, inner_steps
    )
    
    # Evaluate on test trajectory
    test_loss, (policy_loss, value_loss) = compute_inner_loss(
        adapted_policy_params, adapted_value_params,
        policy_apply, value_apply, test_traj
    )
    return test_loss, (policy_loss, value_loss)

# ==============================================================================
# TASK SAMPLING
# ==============================================================================

def sample_task(env: DynamicSpectrumEnv, key: chex.PRNGKey) -> DynamicSpectrumEnv:
    """Sample a task by varying environment parameters"""
    keys = jax.random.split(key, 4)
    
    # Interference variation
    variation_interference = jax.random.uniform(keys[0], (), minval=0.8, maxval=1.2)
    new_max_interference = env.max_interference * variation_interference

    # Fading variation
    fading_variation = jax.random.uniform(keys[1], (), minval=0.9, maxval=1.1)
    new_fading_coherence = jnp.clip(env.fading_coherence * fading_variation, 0.5, 0.99)
    
    # QoS requirement variation
    latency_variation = jax.random.uniform(keys[2], (), minval=0.9, maxval=1.1)
    new_max_latency = env.max_latency * latency_variation
    
    # SINR requirement variation
    sinr_variation = jax.random.uniform(keys[3], (), minval=0.95, maxval=1.05)
    new_min_sinr = env.min_sinr * sinr_variation
    
   
    new_max_power = env.max_power
    new_power_levels = env.power_levels
    
    # Preserve bandwidth and noise figure
    current_bandwidth_hz = getattr(env, 'bandwidth_hz', BANDWIDTH_HZ)
    current_noise_figure_db = getattr(env, 'noise_figure_db', NOISE_FIGURE_DB)

    try:
        new_env = DynamicSpectrumEnv(
            num_bs=env.num_bs,
            num_users=env.num_users,
            num_bands=env.num_bands,
            max_steps=env.max_steps,
            max_latency=new_max_latency,
            max_power=new_max_power,
            num_power_levels=env.num_power_levels,
            power_levels=new_power_levels,
            fading_coherence=new_fading_coherence,
            max_interference=new_max_interference,
            min_sinr=new_min_sinr
        )
        
        # Preserve important attributes
        new_env.bandwidth_hz = current_bandwidth_hz
        new_env.noise_figure_db = current_noise_figure_db
        new_env.thermal_noise_dbm_hz = getattr(env, 'thermal_noise_dbm_hz', -174.0)
        
        return new_env
        
    except Exception as e:
        print(f"Error creating task environment: {e}")
        return env

# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def train_maml(
    env: Any, policy_params: Any, value_params: Any, policy_apply: Callable, value_apply: Callable,
    num_tasks: int, inner_lr: float, inner_steps: int, meta_lr: float, num_iterations: int,
    dim: int, key: jnp.ndarray, eval_interval: int = 10, num_eval_tasks: int = 5,
    wandb_project: str = "maml-training", wandb_name: str = None, use_wandb: bool = True,
    rollout_len_config: int = ROLLOUT_LENGTH
) -> Tuple[Tuple[Any,Any], Dict[str, list]]:
    """Train MAML with simplified approach similar to recurrent model"""

    params = (policy_params, value_params)
    obs_norm_state = init_obs_normalizer(dim)
    reward_norm_state = init_reward_normalizer()
    
    # Optimizer setup
    lr_schedule = optax.exponential_decay(
        init_value=meta_lr * 0.5,
        transition_steps=max(1, num_iterations // 20),
        decay_rate=0.98
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(lr_schedule, eps=1e-8)
    )
    opt_state = optimizer.init(params)

    # History tracking
    meta_losses_hist = []
    eval_avg_pre_rewards_hist, eval_avg_post_rewards_hist, eval_avg_reward_improvements_hist = [], [], []
    eval_avg_pre_sinrs_hist, eval_avg_post_sinrs_hist, eval_avg_sinr_improvements_hist = [], [], []
    eval_avg_pre_qoss_hist, eval_avg_post_qoss_hist, eval_avg_qos_improvements_hist = [], [], []

    if use_wandb:
        config_wandb = {
            "num_tasks_per_meta_batch": num_tasks, "inner_lr": inner_lr, "inner_steps": inner_steps,
            "meta_lr": meta_lr, "num_iterations": num_iterations, "obs_dim": dim,
            "eval_interval": eval_interval, "num_eval_tasks": num_eval_tasks,
            "rollout_len": rollout_len_config
        }
        wandb.init(project=wandb_project, name=wandb_name, config=config_wandb)

    successful_iterations = 0
    consecutive_failures = 0
    
    # Main training loop
    for iteration in range(num_iterations):
        if iteration > 0 and iteration % 10 == 0: 
            print(f"DEBUG: Clearing JAX JIT caches at iteration {iteration}")
            jax.clear_caches()
        
        key, iter_key = jax.random.split(key)
        task_keys = jax.random.split(iter_key, num_tasks)
        
        # Initialize gradients
        accumulated_grads = jax.tree_map(jnp.zeros_like, params)
        total_meta_loss = 0.0
        num_successful_tasks = 0
        
        # Process each task
        for task_idx in range(num_tasks):
            try:
                task_key = task_keys[task_idx]
                key_task, key_train, key_test = jax.random.split(task_key, 3)
                
                # Sample task environment
                task_env = sample_task(env, key_task)
                
                # Sample training trajectory
                train_traj = sample_trajectories_simple(
                    task_env, params[0], params[1], policy_apply, value_apply, key_train,
                    obs_norm_state, reward_norm_state, num_steps=rollout_len_config
                )
                
                if train_traj['observations'].shape[0] == 0:
                    print(f"Task {task_idx}: Empty training trajectory")
                    continue
                    
                # Prepare trajectory with GAE
                train_traj_prepared = prepare_trajectory(train_traj, params[1], value_apply)
                
                # Adapt policy on training trajectory
                adapted_policy_params, adapted_value_params = inner_adaptation(
                    params[0], params[1], policy_apply, value_apply,
                    train_traj_prepared, inner_lr, inner_steps
                )
# Sample test trajectory with adapted parameters
                test_traj = sample_trajectories_simple(
                    task_env, adapted_policy_params, adapted_value_params, 
                    policy_apply, value_apply, key_test,
                    obs_norm_state, reward_norm_state, num_steps=rollout_len_config
                )
                
                if test_traj['observations'].shape[0] == 0:
                    print(f"Task {task_idx}: Empty test trajectory")
                    continue
                    
                # Prepare test trajectory
                test_traj_prepared = prepare_trajectory(test_traj, adapted_value_params, value_apply)
                
                # Compute meta-objective and gradients
                def meta_loss_fn(p):
                    loss, _ = compute_meta_objective(
                        p, train_traj_prepared, test_traj_prepared,
                        policy_apply, value_apply, inner_lr, inner_steps
                    )
                    return loss
                
                task_meta_loss, task_grads = jax.value_and_grad(meta_loss_fn)(params)
                
                # Check if loss and gradients are valid
                if not jnp.isfinite(task_meta_loss):
                    print(f"Task {task_idx}: Non-finite loss")
                    continue
                    
                # Accumulate gradients
                accumulated_grads = jax.tree_map(
                    lambda acc, new: acc + new, 
                    accumulated_grads, task_grads
                )
                total_meta_loss += task_meta_loss
                num_successful_tasks += 1
                
                # Update normalizers with task data
                if train_traj['observations'].shape[0] > 0:
                    obs_norm_state = update_obs_normalizer(obs_norm_state, train_traj['observations'])
                    if 'raw_rewards' in train_traj and train_traj['raw_rewards'].shape[0] > 0:
                        reward_norm_state = update_reward_normalizer(reward_norm_state, train_traj['raw_rewards'])
                        
                if test_traj['observations'].shape[0] > 0:
                    obs_norm_state = update_obs_normalizer(obs_norm_state, test_traj['observations'])
                    if 'raw_rewards' in test_traj and test_traj['raw_rewards'].shape[0] > 0:
                        reward_norm_state = update_reward_normalizer(reward_norm_state, test_traj['raw_rewards'])
                        
            except Exception as e:
                print(f"Error in task {task_idx}: {str(e)[:100]}")
                continue
        
        # Apply meta-update if we have successful tasks
        if num_successful_tasks > 0:
            avg_meta_loss = total_meta_loss / num_successful_tasks
            meta_losses_hist.append(float(avg_meta_loss))
            
            # Average gradients
            avg_grads = jax.tree_map(
                lambda g: g / num_successful_tasks, 
                accumulated_grads
            )
            
            # Check gradient norm
            grad_norm = optax.global_norm(avg_grads)
            if jnp.isfinite(grad_norm) and grad_norm < 100.0:
                updates, opt_state = optimizer.update(avg_grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                consecutive_failures = 0
                successful_iterations += 1
            else:
                print(f"Warning: Extreme gradient norm {grad_norm} at iteration {iteration}")
                consecutive_failures += 1
        else:
            meta_losses_hist.append(float('nan'))
            consecutive_failures += 1
            print(f"Warning: Iteration {iteration} had no successful training tasks.")
            
        if consecutive_failures > 50:
            print(f"Early stopping due to {consecutive_failures} consecutive failures")
            break
            
        # Logging
        log_payload = {
            'meta_loss': meta_losses_hist[-1] if meta_losses_hist else 0.0,
            'successful_tasks_ratio': num_successful_tasks / num_tasks if num_tasks > 0 else 0.0,
            'iteration': iteration
        }
        
        # Evaluation
        if iteration % eval_interval == 0:
            key, eval_key = jax.random.split(key)
            eval_task_keys = jax.random.split(eval_key, num_eval_tasks)
            
            pre_rewards, post_rewards = [], []
            pre_sinrs, post_sinrs = [], []
            pre_qoss, post_qoss = [], []
            
            for eval_idx in range(num_eval_tasks):
                try:
                    eval_task_key = eval_task_keys[eval_idx]
                    key_task, key_pre, key_adapt, key_post = jax.random.split(eval_task_key, 4)
                    
                    # Sample evaluation task
                    eval_task_env = sample_task(env, key_task)
                    
                    # Pre-adaptation trajectory
                    pre_traj = sample_trajectories_simple(
                        eval_task_env, params[0], params[1], policy_apply, value_apply, key_pre,
                        obs_norm_state, reward_norm_state, num_steps=rollout_len_config
                    )
                    
                    if pre_traj['observations'].shape[0] > 0:
                        pre_rewards.append(jnp.mean(pre_traj["rewards"]))
                        pre_sinrs.append(jnp.mean(pre_traj["sinr_violations"]))
                        pre_qoss.append(jnp.mean(pre_traj["qos_violations"]))
                        
                    # Adapt to evaluation task
                    adapt_traj = sample_trajectories_simple(
                        eval_task_env, params[0], params[1], policy_apply, value_apply, key_adapt,
                        obs_norm_state, reward_norm_state, num_steps=rollout_len_config
                    )
                    
                    if adapt_traj['observations'].shape[0] > 0:
                        adapt_traj_prepared = prepare_trajectory(adapt_traj, params[1], value_apply)
                        adapted_p, adapted_v = inner_adaptation(
                            params[0], params[1], policy_apply, value_apply,
                            adapt_traj_prepared, inner_lr, inner_steps
                        )
                    else:
                        adapted_p, adapted_v = params[0], params[1]
                        
                    # Post-adaptation trajectory
                    post_traj = sample_trajectories_simple(
                        eval_task_env, adapted_p, adapted_v, policy_apply, value_apply, key_post,
                        obs_norm_state, reward_norm_state, num_steps=rollout_len_config
                    )
                    
                    if post_traj['observations'].shape[0] > 0:
                        post_rewards.append(jnp.mean(post_traj["rewards"]))
                        post_sinrs.append(jnp.mean(post_traj["sinr_violations"]))
                        post_qoss.append(jnp.mean(post_traj["qos_violations"]))
                        
                except Exception as e:
                    print(f"Error in evaluation task {eval_idx}: {e}")
                    continue
                    
            # Aggregate evaluation metrics
            if pre_rewards:
                avg_pre_r = jnp.mean(jnp.array(pre_rewards))
                avg_post_r = jnp.mean(jnp.array(post_rewards)) if post_rewards else avg_pre_r
                improvement = avg_post_r - avg_pre_r
                
                eval_avg_pre_rewards_hist.append(float(avg_pre_r))
                eval_avg_post_rewards_hist.append(float(avg_post_r))
                eval_avg_reward_improvements_hist.append(float(improvement))
                
                log_payload.update({
                    "eval_avg_pre_reward": float(avg_pre_r),
                    "eval_avg_post_reward": float(avg_post_r),
                    "eval_avg_reward_improvement": float(improvement)
                })
                
            if pre_sinrs:
                avg_pre_s = jnp.mean(jnp.array(pre_sinrs))
                avg_post_s = jnp.mean(jnp.array(post_sinrs)) if post_sinrs else avg_pre_s
                improvement_s = avg_pre_s - avg_post_s  # Lower is better
                
                eval_avg_pre_sinrs_hist.append(float(avg_pre_s))
                eval_avg_post_sinrs_hist.append(float(avg_post_s))
                eval_avg_sinr_improvements_hist.append(float(improvement_s))
                
                log_payload.update({
                    "eval_avg_pre_sinr_violation": float(avg_pre_s),
                    "eval_avg_post_sinr_violation": float(avg_post_s),
                    "eval_avg_sinr_improvement": float(improvement_s)
                })
                
            if pre_qoss:
                avg_pre_q = jnp.mean(jnp.array(pre_qoss))
                avg_post_q = jnp.mean(jnp.array(post_qoss)) if post_qoss else avg_pre_q
                improvement_q = avg_pre_q - avg_post_q  # Lower is better
                
                eval_avg_pre_qoss_hist.append(float(avg_pre_q))
                eval_avg_post_qoss_hist.append(float(avg_post_q))
                eval_avg_qos_improvements_hist.append(float(improvement_q))
                
                log_payload.update({
                    "eval_avg_pre_qos_violation": float(avg_pre_q),
                    "eval_avg_post_qos_violation": float(avg_post_q),
                    "eval_avg_qos_improvement": float(improvement_q)
                })
                
            # Print progress
            pre_r = log_payload.get('eval_avg_pre_reward', 0.0)
            post_r = log_payload.get('eval_avg_post_reward', 0.0)
            improvement_pct = ((post_r - pre_r) / abs(pre_r) * 100) if pre_r != 0 else 0.0
            
            print(f"[Iter {iteration:3d}] "
                  f"meta_loss={log_payload.get('meta_loss', 0.0):6.3f} | "
                  f"success_rate={log_payload.get('successful_tasks_ratio', 0.0):.2f} | "
                  f"pre_r={pre_r:6.3f} post_r={post_r:6.3f} | "
                  f"gain={improvement_pct:+5.2f}%")
        else:
            print(f"[Iter {iteration:3d}] "
                  f"meta_loss={log_payload.get('meta_loss', 0.0):6.3f} | "
                  f"success_rate={log_payload.get('successful_tasks_ratio', 0.0):.2f} | "
                  f"training...")
                  
        if use_wandb:
            wandb.log(log_payload, step=iteration)
            
    if use_wandb:
        wandb.finish()
        
    history = {
        "meta_losses": meta_losses_hist,
        "avg_pre_rewards": eval_avg_pre_rewards_hist,
        "avg_post_rewards": eval_avg_post_rewards_hist,
        "avg_reward_improvements": eval_avg_reward_improvements_hist,
        "avg_pre_sinrs": eval_avg_pre_sinrs_hist,
        "avg_post_sinrs": eval_avg_post_sinrs_hist,
        "avg_sinr_improvements": eval_avg_sinr_improvements_hist,
        "avg_pre_qoss": eval_avg_pre_qoss_hist,
        "avg_post_qoss": eval_avg_post_qoss_hist,
        "avg_qos_improvements": eval_avg_qos_improvements_hist,
        "training_stats": {
            "successful_iterations": successful_iterations,
            "total_iterations": iteration + 1,
            "final_consecutive_failures": consecutive_failures
        }
    }
    return params, history
