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
import chex

tfd = tfp.distributions

# Constants for safer task sampling
BANDWIDTH_HZ = 10e6
NOISE_FIGURE_DB = 7.0

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
        x = hk.Flatten()(obs)
        x = hk.Linear(self.hidden_dim)(x)
        x = jax.nn.relu(x)

        if len(x.shape) == 1:
            x = x[None, :]

        lstm = hk.LSTM(self.lstm_hidden_dim)
        if hidden_state is None:
            batch_size = x.shape[0]
            hidden_state = lstm.initial_state(batch_size)
        else:
            if hidden_state.hidden.ndim == 1:
                hidden_state = hk.LSTMState(
                    hidden=hidden_state.hidden[None, :],
                    cell=hidden_state.cell[None, :]
                )

        x, new_hidden_state = lstm(x, hidden_state)

        if new_hidden_state.hidden.ndim == 2 and new_hidden_state.hidden.shape[0] == 1:
            new_hidden_state = hk.LSTMState(
                hidden=new_hidden_state.hidden.squeeze(0),
                cell=new_hidden_state.cell.squeeze(0)
            )

        if x.shape[0] == 1:
            x = x.squeeze(0)

        # More conservative logit initialization
        logits = hk.Linear(
            self.num_bs * self.num_bands * self.num_power_levels,
            w_init=hk.initializers.TruncatedNormal(0.01),  # Smaller initialization
            b_init=hk.initializers.Constant(0.0)
        )(x)
        logits = logits.reshape(-1, self.num_bs * self.num_bands, self.num_power_levels)
        return logits, new_hidden_state


class ValueNetwork(hk.Module):
    def __init__(self, hidden_dim=64, num_blocks=2):  # Reduced complexity
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
    # Much more conservative clipping to prevent extreme distributions
    logits = jnp.clip(logits, -3.0, 3.0)
    return tfd.Categorical(logits=logits), new_state

def value_fn(obs):
    return ValueNetwork()(obs)

def make_networks(num_bs, num_bands, num_power_levels):
    policy = hk.without_apply_rng(hk.transform(
        lambda obs, hidden_state: recurrent_policy_fn(obs, hidden_state, num_bs, num_bands, num_power_levels)
    ))
    value = hk.without_apply_rng(hk.transform(value_fn))
    return policy, value


ObsNormalizerState = tuple

def init_obs_normalizer(obs_dim: int) -> ObsNormalizerState:
    return (
        jnp.zeros(obs_dim), jnp.ones(obs_dim), jnp.array(0),  
        jnp.array(1e-4), jnp.array(50)  
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
    normed = (obs - mean) / jnp.sqrt(var + eps + 1e-6)
    normed = jnp.clip(normed, -3.0, 3.0)  
    do_norm = count >= min_count
    return jnp.where(do_norm, normed, jnp.clip(obs, -3.0, 3.0)) 

RewardNormalizerState = tuple

def init_reward_normalizer() -> RewardNormalizerState:
    return (
        jnp.array(-300.0),  
        jnp.array(100.0),   
        jnp.array(0),
        jnp.array(1e-4), 
        jnp.array(200)      
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
    do_norm = count >= min_count
    
    
    if count < min_count:
        return rewards / 150.0  
    
    normalized = (rewards - mean) / jnp.sqrt(var + eps + 1e-6)
    
    return jnp.clip(normalized, -50.0, 50.0)  



def sample_trajectories(
    task_env: Any, policy_params: Any, value_params: Any, policy_apply: Any,
    value_apply: Any, key: jnp.ndarray, obs_norm_state: ObsNormalizerState,
    reward_norm_state: RewardNormalizerState, num_steps: int = 10
) -> dict:
    
    try:
        state, timestep = task_env.reset(key)
        initial_obs = flatten_state(state)
        
        
        if not jnp.all(jnp.isfinite(initial_obs)):
            print(f"Warning: Invalid initial observation detected")
            return _create_empty_trajectory()
            
        initial_norm_obs = normalize_obs(obs_norm_state, initial_obs)
        _, initial_hidden_state_for_traj = policy_apply(policy_params, initial_norm_obs, None)

    except Exception as e:
        print(f"Error in trajectory initialization: {e}")
        return _create_empty_trajectory()

    def step_fn(carry, _):
        env_state, current_hidden_s, current_norm_o, current_key = carry
        
        try:
            key_step, action_key_step = jax.random.split(current_key)
            
            # Get action distribution and next hidden state
            action_dist, next_hidden_s = policy_apply(policy_params, current_norm_o, current_hidden_s)
            action_sample = action_dist.sample(seed=action_key_step)
            log_prob_sample = action_dist.log_prob(action_sample)
            
            
            if log_prob_sample.ndim > 0:
                log_prob_sample = jnp.sum(log_prob_sample)
                
           
            if not jnp.isfinite(log_prob_sample):
                log_prob_sample = jnp.array(-10.0)  
            
            value_estimate = value_apply(value_params, current_norm_o).squeeze()
            if not jnp.isfinite(value_estimate):
                value_estimate = jnp.array(0.0)
                
            action_flat_for_env = action_sample.reshape(-1)
            
           
            next_env_state, next_env_timestep = task_env.step(env_state, action_flat_for_env)
            raw_reward = next_env_timestep.reward
            
            
            if not jnp.isfinite(raw_reward):
                raw_reward = jnp.array(-0.1)  
                
            norm_reward_val = normalize_reward(reward_norm_state, raw_reward)
            
            
            try:
                sinr_vals = task_env._compute_sinr(next_env_state)
                user_best_sinr = jnp.max(sinr_vals, axis=1)
                sinr_violation_count = jnp.sum(user_best_sinr < task_env.min_sinr)
                qos_violation_count = jnp.sum(next_env_state.qos_metrics[:, 0] > task_env.max_latency)
            except Exception:
                sinr_violation_count = jnp.array(0.0)
                qos_violation_count = jnp.array(0.0)
            
            next_flat_obs = flatten_state(next_env_state)
            
            
            if not jnp.all(jnp.isfinite(next_flat_obs)):
                
                done_flag = jnp.array(True)
                next_norm_o = current_norm_o  
            else:
                next_norm_o = normalize_obs(obs_norm_state, next_flat_obs)
                done_flag = next_env_timestep.last()
            
            new_carry = (next_env_state, next_hidden_s, next_norm_o, key_step)
            outputs_step = (
                current_norm_o, action_sample, norm_reward_val, value_estimate, done_flag,
                log_prob_sample, 
                sinr_violation_count, qos_violation_count, raw_reward
            )
            return new_carry, outputs_step
            
        except Exception as e:
            done_flag = jnp.array(True)
            safe_outputs = (
                current_norm_o, 
                jnp.zeros_like(action_sample) if 'action_sample' in locals() else jnp.zeros((env_state.spectrum_alloc.shape[0] * env_state.spectrum_alloc.shape[1],), dtype=jnp.int32),
                jnp.array(-1.0), jnp.array(0.0), done_flag, jnp.array(-10.0),
                jnp.array(0.0), jnp.array(0.0), jnp.array(-1.0)
            )
            return carry, safe_outputs

    try:
        scan_initial_carry = (state, initial_hidden_state_for_traj, initial_norm_obs, key)
        final_carry, scan_outputs = jax.lax.scan(step_fn, scan_initial_carry, xs=None, length=num_steps)

        (observations_all, actions_all, norm_rewards_all, values_all, dones_all, log_probs_all,
         sinr_violations_all, qos_violations_all, raw_rewards_all) = scan_outputs
        
        final_norm_obs_for_gae = final_carry[2]
        final_value_for_gae = value_apply(value_params, final_norm_obs_for_gae).squeeze()
        
        if not jnp.all(jnp.isfinite(observations_all)):
            print("Warning: Invalid observations in trajectory")
            return _create_empty_trajectory()

        return {
            'observations': observations_all, 'actions': actions_all, 'rewards': norm_rewards_all,
            'raw_rewards': raw_rewards_all, 'values': values_all, 'dones': dones_all,
            'log_probs': log_probs_all,
            'initial_hidden_state_for_ppo': initial_hidden_state_for_traj, 
            'final_value': final_value_for_gae,
            'final_hidden_state': final_carry[1], 
            'sinr_violations': sinr_violations_all, 'qos_violations': qos_violations_all,
        }
    except Exception as e:
        print(f"Error in trajectory scan: {e}")
        return _create_empty_trajectory()


def _create_empty_trajectory():
    """Helper function to create empty trajectory for error cases"""
    return {
        'observations': jnp.array([]).reshape(0, 70),  
        'actions': jnp.array([]).reshape(0, 15),  
        'rewards': jnp.array([]),
        'raw_rewards': jnp.array([]),
        'values': jnp.array([]),
        'dones': jnp.array([], dtype=bool),
        'log_probs': jnp.array([]),
        'initial_hidden_state_for_ppo': None,
        'final_value': jnp.array(0.0),
        'final_hidden_state': None,
        'sinr_violations': jnp.array([]),
        'qos_violations': jnp.array([])
    }

def sample_trajectories_simple(
    task_env: Any, policy_params: Any, value_params: Any, policy_apply: Any,
    value_apply: Any, key: jnp.ndarray, obs_norm_state: ObsNormalizerState,
    reward_norm_state: RewardNormalizerState, num_steps: int = 10
) -> dict:
    """Simple non-JAX trajectory sampling for debugging"""
    
    
    state, timestep = task_env.reset(key)
    initial_obs = flatten_state(state)
    initial_norm_obs = normalize_obs(obs_norm_state, initial_obs)
    _, hidden_state = policy_apply(policy_params, initial_norm_obs, None)
    
    
    observations = []
    actions = []
    rewards = []
    raw_rewards = []
    values = []
    dones = []
    log_probs = []
    sinr_violations = []
    qos_violations = []
    
    current_obs = initial_norm_obs
    current_state = state
    current_hidden = hidden_state
    
    for step in range(num_steps):
        key, step_key = jax.random.split(key)
        
        # Get action
        action_dist, current_hidden = policy_apply(policy_params, current_obs, current_hidden)
        action = action_dist.sample(seed=step_key)
        log_prob = action_dist.log_prob(action)
        if log_prob.ndim > 0:
            log_prob = jnp.sum(log_prob)
            
        # Get value
        value = value_apply(value_params, current_obs).squeeze()
        
        # Environment step
        action_flat = action.reshape(-1)
        next_state, next_timestep = task_env.step(current_state, action_flat)
        raw_reward = next_timestep.reward
        
        # Scale reward
        norm_reward = raw_reward / 100.0
        
        # Get next observation
        next_obs = flatten_state(next_state)
        next_norm_obs = normalize_obs(obs_norm_state, next_obs)
        
        # Calculate violations
        sinr_vals = task_env._compute_sinr(next_state)
        user_best_sinr = jnp.max(sinr_vals, axis=1)
        sinr_violation_count = jnp.sum(user_best_sinr < task_env.min_sinr)
        qos_violation_count = jnp.sum(next_state.qos_metrics[:, 0] > task_env.max_latency)
        
        # Store data
        observations.append(current_obs)
        actions.append(action)
        rewards.append(norm_reward)
        raw_rewards.append(raw_reward)
        values.append(value)
        dones.append(next_timestep.last())
        log_probs.append(log_prob)
        sinr_violations.append(sinr_violation_count)
        qos_violations.append(qos_violation_count)
        
        # Update for next step
        current_obs = next_norm_obs
        current_state = next_state
        
        if next_timestep.last():
            break
    
    # Convert to arrays
    final_value = value_apply(value_params, current_obs).squeeze()
    
    return {
        'observations': jnp.array(observations),
        'actions': jnp.array(actions),
        'rewards': jnp.array(rewards),
        'raw_rewards': jnp.array(raw_rewards),
        'values': jnp.array(values),
        'dones': jnp.array(dones),
        'log_probs': jnp.array(log_probs),
        'initial_hidden_state_for_ppo': hidden_state,
        'final_value': final_value,
        'final_hidden_state': current_hidden,
        'sinr_violations': jnp.array(sinr_violations),
        'qos_violations': jnp.array(qos_violations),
    }



def compute_gae(rewards, values, dones, final_value, gamma=0.99, lambda_=0.95):
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
    if traj['observations'].shape[0] == 0:
        return {**traj, 'returns': jnp.array([]), 'advantages': jnp.array([])}
    returns, advantages = compute_gae(
        traj['rewards'], traj['values'], traj['dones'], traj['final_value']
    )
    return {**traj, 'returns': returns, 'advantages': advantages}


def compute_ppo_loss(
    policy_params, value_params, policy_apply, value_apply, traj,
    clip_ratio=0.15, value_coeff=0.3, entropy_coeff=0.01  # More conservative values
):
    obs_seq = traj['observations']
    actions_seq = traj['actions']
    returns_seq = traj['returns']
    advantages_seq = traj['advantages']
    old_log_probs_seq = traj['log_probs']
    dones_seq = traj['dones']

    if obs_seq.shape[0] == 0:
        zero_loss_scalar = jnp.array(0.0, dtype=jnp.float32)
        return zero_loss_scalar, (zero_loss_scalar, zero_loss_scalar, zero_loss_scalar)

    initial_hidden_state_for_scan = traj['initial_hidden_state_for_ppo'] 

    def ppo_scan_step(hidden_state_carry, scan_inputs_t):
        obs_t, action_t, old_log_prob_t, advantage_t, return_t = scan_inputs_t
        
        try:
            action_dist_t, next_hidden_state_carry = policy_apply(policy_params, obs_t, hidden_state_carry)
            current_log_prob_t = action_dist_t.log_prob(action_t)
            entropy_t = action_dist_t.entropy()
            value_t = value_apply(value_params, obs_t).squeeze()

            # Handle multi-dimensional outputs
            if current_log_prob_t.ndim > 0:
                current_log_prob_t = jnp.sum(current_log_prob_t)
            if old_log_prob_t.ndim > 0:
                old_log_prob_t = jnp.sum(old_log_prob_t)
            if entropy_t.ndim > 0:
                entropy_t = jnp.mean(entropy_t)

            
            log_ratio_t = current_log_prob_t - old_log_prob_t
            log_ratio_t = jnp.clip(log_ratio_t, -5.0, 5.0)  
            ratio_t = jnp.exp(log_ratio_t)

            # PPO clipped objective
            surr1_t = ratio_t * advantage_t
            surr2_t = jnp.clip(ratio_t, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage_t
            policy_loss_term_t = -jnp.minimum(surr1_t, surr2_t)
            
            value_loss_term_t = jnp.square(return_t - value_t)
            entropy_bonus_term_t = -entropy_t

            policy_loss_term_t = jnp.where(jnp.isfinite(policy_loss_term_t), policy_loss_term_t, 0.0)
            value_loss_term_t = jnp.where(jnp.isfinite(value_loss_term_t), value_loss_term_t, 0.0)
            entropy_bonus_term_t = jnp.where(jnp.isfinite(entropy_bonus_term_t), entropy_bonus_term_t, 0.0)

            return next_hidden_state_carry, (policy_loss_term_t, value_loss_term_t, entropy_bonus_term_t)
            
        except Exception as e:
            return hidden_state_carry, (jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))

    scan_inputs_stacked = (obs_seq, actions_seq, old_log_probs_seq, advantages_seq, returns_seq)
    
    try:
        final_hidden_state, (policy_loss_terms, value_loss_terms, entropy_bonus_terms) = jax.lax.scan(
            f=ppo_scan_step, init=initial_hidden_state_for_scan, xs=scan_inputs_stacked
        )
    except Exception as e:
        print(f"Error in PPO loss scan: {e}")
        zero_loss = jnp.array(0.0)
        return zero_loss, (zero_loss, zero_loss, zero_loss)

    # Calculate valid steps and apply masking
    actual_seq_len = obs_seq.shape[0]
    indices = jnp.arange(actual_seq_len)
    first_true_idx_or_len = jnp.min(jnp.where(dones_seq, indices, actual_seq_len))
    num_valid_steps = jnp.minimum(first_true_idx_or_len + 1, actual_seq_len)

    valid_step_mask = jnp.arange(actual_seq_len) < num_valid_steps
    valid_step_mask_float = valid_step_mask.astype(jnp.float32)

    # Apply masking and compute mean losses
    sum_policy_loss = jnp.sum(policy_loss_terms * valid_step_mask_float)
    sum_value_loss = jnp.sum(value_loss_terms * valid_step_mask_float)
    sum_entropy_bonus = jnp.sum(entropy_bonus_terms * valid_step_mask_float)

    count_valid_steps = jnp.sum(valid_step_mask_float)
    safe_count_valid_steps = jnp.maximum(count_valid_steps, 1.0)

    mean_policy_loss = sum_policy_loss / safe_count_valid_steps
    mean_value_loss = sum_value_loss / safe_count_valid_steps
    mean_entropy_bonus = sum_entropy_bonus / safe_count_valid_steps

    # Apply loss only when there are valid steps
    mean_policy_loss = jnp.where(count_valid_steps > 0, mean_policy_loss, 0.0)
    mean_value_loss = jnp.where(count_valid_steps > 0, mean_value_loss, 0.0)
    mean_entropy_bonus = jnp.where(count_valid_steps > 0, mean_entropy_bonus, 0.0)

    mean_policy_loss = jnp.clip(mean_policy_loss, -10.0, 10.0)
    mean_value_loss = jnp.clip(mean_value_loss, 0.0, 10.0)
    mean_entropy_bonus = jnp.clip(mean_entropy_bonus, -2.0, 2.0)

    total_loss = mean_policy_loss + value_coeff * mean_value_loss + entropy_coeff * mean_entropy_bonus
    
    # Final safety check
    total_loss = jnp.where(jnp.isfinite(total_loss), total_loss, 0.0)
    
    return total_loss, (mean_policy_loss, mean_value_loss, mean_entropy_bonus)


def ppo_inner_adaptation(
    policy_params, value_params, policy_apply, value_apply, traj,
    inner_lr, inner_steps, clip_ratio=0.1
):
    if traj['observations'].shape[0] == 0:
        return policy_params, value_params
        
    initial_params_for_adaptation = (policy_params, value_params)
    
    def adaptation_fori_body(loop_idx, current_params_tuple_fori):
        curr_policy_params_f, curr_value_params_f = current_params_tuple_fori
        
        def loss_fn_for_grad_f(params_for_loss_f):
            p_params_f, v_params_f = params_for_loss_f
            loss_val_f, _ = compute_ppo_loss(
                p_params_f, v_params_f, policy_apply, value_apply, traj, clip_ratio
            )
            return loss_val_f
            
        try:
            loss_val, grads_tuple_f = jax.value_and_grad(loss_fn_for_grad_f)((curr_policy_params_f, curr_value_params_f))
            policy_grads_f, value_grads_f = grads_tuple_f
            
            
            max_grad_norm = 0.5  
            
            policy_grad_norm = optax.global_norm(policy_grads_f)
            value_grad_norm = optax.global_norm(value_grads_f)
            
            policy_clip_factor = jnp.minimum(1.0, max_grad_norm / (policy_grad_norm + 1e-8))
            value_clip_factor = jnp.minimum(1.0, max_grad_norm / (value_grad_norm + 1e-8))
            
            policy_grads_f = jax.tree_map(lambda g: g * policy_clip_factor, policy_grads_f)
            value_grads_f = jax.tree_map(lambda g: g * value_clip_factor, value_grads_f)
            
            # Apply gradients with smaller learning rate
            effective_lr = inner_lr * 0.5  
            new_policy_params_f = jax.tree_map(lambda p, g: p - effective_lr * g, curr_policy_params_f, policy_grads_f)
            new_value_params_f = jax.tree_map(lambda p, g: p - effective_lr * g, curr_value_params_f, value_grads_f)
            
            return (new_policy_params_f, new_value_params_f)
            
        except Exception as e:
            return (curr_policy_params_f, curr_value_params_f)

    try:
        final_adapted_params_tuple = lax.fori_loop(
            0, inner_steps, adaptation_fori_body, initial_params_for_adaptation
        )
        return final_adapted_params_tuple
    except Exception as e:
        print(f"Error in inner adaptation: {e}")
        return initial_params_for_adaptation


@partial(jax.jit, static_argnums=(3,4,5,6,7))
def compute_meta_objective(
    init_params, train_traj, test_traj,
    policy_apply, value_apply, inner_lr, inner_steps, clip_ratio=0.1
):
    init_policy_params, init_value_params = init_params

    # Perform inner adaptation on training trajectory
    adapted_policy_params, adapted_value_params = ppo_inner_adaptation(
        init_policy_params, init_value_params,
        policy_apply, value_apply, train_traj,
        inner_lr, inner_steps, clip_ratio
    )

    # Evaluate on test trajectory using adapted parameters
    test_loss, (policy_loss, value_loss, entropy_loss_aux) = compute_ppo_loss(
        adapted_policy_params, adapted_value_params,
        policy_apply, value_apply, test_traj, clip_ratio
    )
    return test_loss, (policy_loss, value_loss, entropy_loss_aux)


def sample_task(env: DynamicSpectrumEnv, key: chex.PRNGKey) -> DynamicSpectrumEnv: 
    keys = jax.random.split(key, 4)  # Reduced variations for stability
    
    variation_interference = jax.random.uniform(keys[0], (), minval=0.8, maxval=1.2)
    new_max_interference = env.max_interference * variation_interference

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
        
        new_env.bandwidth_hz = current_bandwidth_hz
        new_env.noise_figure_db = current_noise_figure_db
        new_env.thermal_noise_dbm_hz = getattr(env, 'thermal_noise_dbm_hz', -174.0)
        
        return new_env
        
    except Exception as e:
        print(f"Error creating task environment: {e}")
        return env


def stack_trajectories(trajs: List[Dict]) -> Dict:
    if not trajs:
        return {
            'observations': jnp.array([]), 'actions': jnp.array([]), 'rewards': jnp.array([]),
            'raw_rewards': jnp.array([]), 'values': jnp.array([]), 'dones': jnp.array([], dtype=bool),
            'log_probs': jnp.array([]), 'initial_hidden_state_for_ppo': [], 
            'final_value': jnp.array([]), 'final_hidden_state': [],
            'sinr_violations': jnp.array([]), 'qos_violations': jnp.array([])
        }
    keys = trajs[0].keys()
    stacked = {}
    for k in keys:
        if k in ['initial_hidden_state_for_ppo', 'final_hidden_state']:
             stacked[k] = [traj[k] for traj in trajs]
        elif isinstance(trajs[0][k], jnp.ndarray):
            try: 
                stacked[k] = jnp.stack([traj[k] for traj in trajs])
            except ValueError:
                print(f"Warning: Could not stack key '{k}' due to inconsistent shapes. Storing as list.")
                stacked[k] = [traj[k] for traj in trajs]
        else: 
            stacked[k] = [traj[k] for traj in trajs] 
    return stacked


def enhanced_logging(iteration, metrics, history):
    """Enhanced logging with trend analysis"""
    if iteration % 20 == 0 and iteration > 0:
        print(f"\n=== Training Summary (Iteration {iteration}) ===")
        
        
        recent_window = min(10, len(history['avg_post_rewards']))
        if recent_window > 1:
            recent_rewards = history['avg_post_rewards'][-recent_window:]
            trend = "↗️" if recent_rewards[-1] > recent_rewards[0] else "↘️"
            print(f"Recent reward trend: {trend} ({recent_rewards[0]:.3f} → {recent_rewards[-1]:.3f})")
        
        pre_r = metrics.get('eval_avg_pre_reward', 0.0)
        post_r = metrics.get('eval_avg_post_reward', 0.0)
        print(f"Current: pre={pre_r:.3f}, post={post_r:.3f}, gain={((post_r-pre_r)/abs(pre_r)*100 if pre_r != 0 else 0):.1f}%")
        print("=" * 50)


def train_recurrent_maml_ppo(
    env: Any, policy_params: Any, value_params: Any, policy_apply: Any, value_apply: Any,
    num_tasks: int, inner_lr: float, inner_steps: int, meta_lr: float, num_iterations: int,
    obs_dim: int, key: jnp.ndarray, clip_ratio: float = 0.1, eval_interval: int = 10,
    num_eval_tasks: int = 5, wandb_project: str = "recurrent-maml-ppo", wandb_name: str = None,
    use_wandb: bool = True, rollout_len: int = 20  
) -> Tuple[Tuple[Any,Any], Dict[str, list]]:

    params = (policy_params, value_params)
    obs_norm_state = init_obs_normalizer(obs_dim)
    reward_norm_state = init_reward_normalizer()

    
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
            "meta_lr": meta_lr, "num_iterations": num_iterations, "obs_dim": obs_dim,
            "clip_ratio": clip_ratio, "eval_interval": eval_interval, "num_eval_tasks": num_eval_tasks,
            "rollout_len": rollout_len
        }
        wandb.init(project=wandb_project, name=wandb_name, config=config_wandb)

    def get_meta_loss_and_grads_for_task(current_params, train_traj_arg, test_traj_arg,
                                         p_apply, v_apply, i_lr, i_steps, c_ratio):
        loss_val, _aux = compute_meta_objective( 
            current_params, train_traj_arg, test_traj_arg,
            p_apply, v_apply, i_lr, i_steps, c_ratio
        )
        return loss_val
    value_and_grad_fn_for_task = jax.value_and_grad(get_meta_loss_and_grads_for_task, argnums=0)

    successful_iterations = 0
    consecutive_failures = 0
    
    # Training loop with enhanced debugging and stability
    for iteration in range(num_iterations):
        if iteration > 0 and iteration % 10 == 0: 
            print(f"DEBUG: Clearing JAX JIT caches at iteration {iteration}")
            jax.clear_caches()
        
        key, iter_key = jax.random.split(key)
        task_keys_for_batch = jax.random.split(iter_key, num_tasks)
        accumulated_meta_grads = None
        total_meta_loss_this_batch = 0.0
        num_successful_tasks_in_batch = 0
        
        
        task_failures = []

        for task_idx in range(num_tasks):
            try:
                task_specific_key = task_keys_for_batch[task_idx]
                key_task_sample, key_train_traj, key_test_traj = jax.random.split(task_specific_key, 3)
                task_env_instance = sample_task(env, key_task_sample)

                train_traj_raw = sample_trajectories_simple(
                    task_env_instance, params[0], params[1], policy_apply, value_apply, key_train_traj,
                    obs_norm_state, reward_norm_state, num_steps=rollout_len
                )
                
                # Debug trajectory validity
                if train_traj_raw['observations'].shape[0] == 0: 
                    task_failures.append(f"Task {task_idx}: Empty training trajectory")
                    continue
                    
                # Check for NaN in observations
                if not jnp.all(jnp.isfinite(train_traj_raw['observations'])):
                    task_failures.append(f"Task {task_idx}: NaN in training observations")
                    continue
                    
                train_traj_prepared = prepare_trajectory(train_traj_raw, params[1], value_apply)
                
                # Check GAE outputs
                if (train_traj_prepared['returns'].shape[0] > 0 and 
                    not jnp.all(jnp.isfinite(train_traj_prepared['returns']))):
                    task_failures.append(f"Task {task_idx}: NaN in returns")
                    continue

                # Use current parameters for test sampling 
                adapted_policy_p_for_test_sampling, adapted_value_p_for_test_sampling = params[0], params[1]

                # Sample test trajectory
                test_traj_raw = sample_trajectories_simple(
                    task_env_instance, adapted_policy_p_for_test_sampling, adapted_value_p_for_test_sampling,
                    policy_apply, value_apply, key_test_traj,
                    obs_norm_state, reward_norm_state, num_steps=rollout_len
                )

                # Debug test trajectory
                if test_traj_raw['observations'].shape[0] == 0: 
                    task_failures.append(f"Task {task_idx}: Empty test trajectory")
                    continue
                    
                if not jnp.all(jnp.isfinite(test_traj_raw['observations'])):
                    task_failures.append(f"Task {task_idx}: NaN in test observations")
                    continue
                    
                test_traj_prepared = prepare_trajectory(test_traj_raw, adapted_value_p_for_test_sampling, value_apply)

                # Compute meta-objective and gradients
                task_meta_loss_val, task_meta_grads = value_and_grad_fn_for_task(
                    params, train_traj_prepared, test_traj_prepared,
                    policy_apply, value_apply, inner_lr, inner_steps, clip_ratio
                )

                # Enhanced validity checks
                if not jnp.isfinite(task_meta_loss_val):
                    task_failures.append(f"Task {task_idx}: Non-finite meta loss: {task_meta_loss_val}")
                    continue
                    
                # Check gradient validity
                grad_is_finite = True
                for grad_tree in [task_meta_grads[0], task_meta_grads[1]]:  
                    grad_leaves = jax.tree_leaves(grad_tree)
                    if not all(jnp.all(jnp.isfinite(leaf)) for leaf in grad_leaves):
                        grad_is_finite = False
                        break
                        
                if not grad_is_finite:
                    task_failures.append(f"Task {task_idx}: Non-finite gradients")
                    continue

                total_meta_loss_this_batch += task_meta_loss_val
                
                if accumulated_meta_grads is None: 
                    accumulated_meta_grads = task_meta_grads
                else: 
                    accumulated_meta_grads = jax.tree_map(
                        lambda acc_g, new_g: acc_g + new_g, 
                        accumulated_meta_grads, task_meta_grads
                    )
                num_successful_tasks_in_batch += 1

                # Update normalizers with current task data
                if train_traj_raw['observations'].shape[0] > 0:
                    obs_norm_state = update_obs_normalizer(obs_norm_state, train_traj_raw['observations'])
                    if 'raw_rewards' in train_traj_raw and train_traj_raw['raw_rewards'].shape[0] > 0:
                         reward_norm_state = update_reward_normalizer(reward_norm_state, train_traj_raw['raw_rewards'])
                         
                if test_traj_raw['observations'].shape[0] > 0:
                    obs_norm_state = update_obs_normalizer(obs_norm_state, test_traj_raw['observations'])
                    if 'raw_rewards' in test_traj_raw and test_traj_raw['raw_rewards'].shape[0] > 0:
                        reward_norm_state = update_reward_normalizer(reward_norm_state, test_traj_raw['raw_rewards'])

            except Exception as e:
                task_failures.append(f"Task {task_idx}: Exception - {str(e)[:100]}")
                continue

        # Enhanced debugging output
        if task_failures and iteration % 5 == 0:  
            print(f"Iteration {iteration} task failures:")
            for failure in task_failures[:3]:  
                print(f"  {failure}")
            if len(task_failures) > 3:
                print(f"  ... and {len(task_failures) - 3} more")

        if num_successful_tasks_in_batch > 0:
            avg_meta_loss_batch = total_meta_loss_this_batch / num_successful_tasks_in_batch
            meta_losses_hist.append(float(avg_meta_loss_batch))
            
            # Average gradients across successful tasks
            mean_meta_grads = jax.tree_map(
                lambda g: g / num_successful_tasks_in_batch, 
                accumulated_meta_grads
            )
            
            # Apply gradients with additional safety checks
            try:
                total_grad_norm = optax.global_norm(mean_meta_grads)
                if jnp.isfinite(total_grad_norm) and total_grad_norm < 100.0:  
                    updates, opt_state = optimizer.update(mean_meta_grads, opt_state, params)
                    params = optax.apply_updates(params, updates)
                    consecutive_failures = 0
                    successful_iterations += 1
                else:
                    print(f"Warning: Extreme gradient norm {total_grad_norm} at iteration {iteration}")
                    meta_losses_hist.append(float('nan'))
                    consecutive_failures += 1
            except Exception as e:
                print(f"Warning: Parameter update failed at iteration {iteration}: {e}")
                meta_losses_hist.append(float('nan'))
                consecutive_failures += 1
        else:
            meta_losses_hist.append(float('nan'))
            consecutive_failures += 1
            print(f"Warning: Iteration {iteration} had no successful training tasks.")

        if consecutive_failures > 50:
            print(f"Early stopping due to {consecutive_failures} consecutive failures")
            break

        log_payload_eval = {
            'meta_loss': meta_losses_hist[-1] if meta_losses_hist else 0.0,
            'successful_tasks_ratio': num_successful_tasks_in_batch / num_tasks if num_tasks > 0 else 0.0,
            'eval_avg_pre_reward': 0.0,
            'eval_avg_post_reward': 0.0,
            'eval_avg_reward_improvement': 0.0
        }

        # Evaluation loop 
        if iteration % eval_interval == 0: 
            log_payload_eval = {
                "iteration": iteration, 
                "meta_loss": meta_losses_hist[-1] if meta_losses_hist and jnp.isfinite(meta_losses_hist[-1]) else float('nan'),
                "successful_tasks_ratio": num_successful_tasks_in_batch / num_tasks,
                "consecutive_failures": consecutive_failures
            }
            
            key, eval_master_key = jax.random.split(key)
            eval_task_keys = jax.random.split(eval_master_key, num_eval_tasks)
            
            current_eval_pre_rewards, current_eval_post_rewards = [], []
            current_eval_pre_sinrs, current_eval_post_sinrs = [], []
            current_eval_pre_qoss, current_eval_post_qoss = [], []

            for eval_task_idx in range(num_eval_tasks):
                try:
                    eval_task_specific_key = eval_task_keys[eval_task_idx]
                    key_eval_task_sample, key_eval_pre_traj, key_eval_adapt, key_eval_post_traj = jax.random.split(eval_task_specific_key, 4)
                    eval_task_env = sample_task(env, key_eval_task_sample)
                    
                    # Pre-adaptation evaluation
                    pre_adapt_traj_eval = sample_trajectories_simple(
                        eval_task_env, params[0], params[1], policy_apply, value_apply, key_eval_pre_traj,
                        obs_norm_state, reward_norm_state, num_steps=rollout_len
                    )
                    
                    if pre_adapt_traj_eval['observations'].shape[0] > 0:
                        current_eval_pre_rewards.append(jnp.mean(pre_adapt_traj_eval["rewards"]))
                        current_eval_pre_sinrs.append(jnp.mean(pre_adapt_traj_eval["sinr_violations"]))
                        current_eval_pre_qoss.append(jnp.mean(pre_adapt_traj_eval["qos_violations"]))

                    # Adaptation
                    adapted_policy_p_eval, adapted_value_p_eval = params[0], params[1]
                    if pre_adapt_traj_eval['observations'].shape[0] > 0:
                        pre_adapt_traj_eval_prepared = prepare_trajectory(pre_adapt_traj_eval, params[1], value_apply)
                        adapted_policy_p_eval, adapted_value_p_eval = ppo_inner_adaptation(
                            params[0], params[1], policy_apply, value_apply,
                            pre_adapt_traj_eval_prepared, inner_lr, inner_steps, clip_ratio
                        )

                    # Post-adaptation evaluation
                    post_adapt_traj_eval = sample_trajectories_simple(
                        eval_task_env, adapted_policy_p_eval, adapted_value_p_eval, policy_apply, value_apply, key_eval_post_traj,
                        obs_norm_state, reward_norm_state, num_steps=rollout_len
                    )
                    
                    if post_adapt_traj_eval['observations'].shape[0] > 0:
                        current_eval_post_rewards.append(jnp.mean(post_adapt_traj_eval["rewards"]))
                        current_eval_post_sinrs.append(jnp.mean(post_adapt_traj_eval["sinr_violations"]))
                        current_eval_post_qoss.append(jnp.mean(post_adapt_traj_eval["qos_violations"]))

                except Exception as e:
                    continue

            # Aggregate evaluation metrics and update history
            if current_eval_pre_rewards:
                apr = jnp.mean(jnp.array(current_eval_pre_rewards))
                apsr = jnp.mean(jnp.array(current_eval_post_rewards)) if current_eval_post_rewards else apr
                asi = apsr - apr
                eval_avg_pre_rewards_hist.append(float(apr))
                eval_avg_post_rewards_hist.append(float(apsr))
                eval_avg_reward_improvements_hist.append(float(asi))
                log_payload_eval.update({
                    "eval_avg_pre_reward": float(apr), 
                    "eval_avg_post_reward": float(apsr), 
                    "eval_avg_reward_improvement": float(asi)
                })
                
            if current_eval_pre_sinrs:
                aps = jnp.mean(jnp.array(current_eval_pre_sinrs))
                aops = jnp.mean(jnp.array(current_eval_post_sinrs)) if current_eval_post_sinrs else aps
                assi = aps - aops  
                eval_avg_pre_sinrs_hist.append(float(aps))
                eval_avg_post_sinrs_hist.append(float(aops))
                eval_avg_sinr_improvements_hist.append(float(assi))
                log_payload_eval.update({
                    "eval_avg_pre_sinr_violation": float(aps), 
                    "eval_avg_post_sinr_violation": float(aops), 
                    "eval_avg_sinr_improvement": float(assi)
                })
                
            if current_eval_pre_qoss:
                apq = jnp.mean(jnp.array(current_eval_pre_qoss))
                apoq = jnp.mean(jnp.array(current_eval_post_qoss)) if current_eval_post_qoss else apq
                asqi = apq - apoq  
                eval_avg_pre_qoss_hist.append(float(apq))
                eval_avg_post_qoss_hist.append(float(apoq))
                eval_avg_qos_improvements_hist.append(float(asqi))
                log_payload_eval.update({
                    "eval_avg_pre_qos_violation": float(apq), 
                    "eval_avg_post_qos_violation": float(apoq), 
                    "eval_avg_qos_improvement": float(asqi)
                })

            # Enhanced logging for evaluation iterations
            pre_reward = log_payload_eval.get('eval_avg_pre_reward', 0.0)
            post_reward = log_payload_eval.get('eval_avg_post_reward', 0.0)
            improvement_pct = ((post_reward - pre_reward) / abs(pre_reward)) * 100 if pre_reward != 0 else 0.0

            print(f"[Iter {iteration:3d}] "
                  f"meta_loss={log_payload_eval.get('meta_loss', 0.0):6.3f} | "
                  f"success_rate={log_payload_eval.get('successful_tasks_ratio', 0.0):.2f} | "
                  f"pre_r={pre_reward:6.3f} post_r={post_reward:6.3f} | "
                  f"gain={improvement_pct:+5.2f}%")
            
            if iteration > 0 and iteration % 20 == 0 and len(eval_avg_post_rewards_hist) >= 2:
                recent_trend = "↗️" if eval_avg_post_rewards_hist[-1] > eval_avg_post_rewards_hist[-2] else "↘️"
                print(f"\n=== Training Summary (Iteration {iteration}) ===")
                print(f"Recent reward trend: {recent_trend} ({eval_avg_post_rewards_hist[0]:.3f} → {eval_avg_post_rewards_hist[-1]:.3f})")
                print(f"Current: pre={pre_reward:.3f}, post={post_reward:.3f}, gain={improvement_pct:.1f}%")
                print("=" * 50)
            
            if use_wandb: 
                wandb.log(log_payload_eval, step=iteration)
        
        else:
            basic_log_payload = {
                "iteration": iteration,
                "meta_loss": meta_losses_hist[-1] if meta_losses_hist and jnp.isfinite(meta_losses_hist[-1]) else float('nan'),
                "successful_tasks_ratio": num_successful_tasks_in_batch / num_tasks,
            }
            
            print(f"[Iter {iteration:3d}] "
                  f"meta_loss={basic_log_payload.get('meta_loss', 0.0):6.3f} | "
                  f"success_rate={basic_log_payload.get('successful_tasks_ratio', 0.0):.2f} | "
                  f"training...")
        
        if use_wandb: 
            wandb.log(log_payload_eval, step=iteration)

    if use_wandb: 
        wandb.finish()
        
    history_data = {
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
    return params, history_data
