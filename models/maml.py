from jumanji.types import TimeStep 
from functools import partial
from typing import Any, Callable, Tuple, Dict
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

# Constants
ROLLOUT_LENGTH = 50 

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

ObsNormalizerState = tuple

def init_obs_normalizer(obs_dim: int) -> ObsNormalizerState:
    return (
        jnp.zeros(obs_dim), jnp.ones(obs_dim) * 10.0, jnp.array(0),
        jnp.array(1e-4), jnp.array(100)
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

def normalize_obs(state, obs: jnp.ndarray) -> jnp.ndarray:
    mean, var, count, eps, min_count = state
    normed = (obs - mean) / jnp.sqrt(var + eps + 1e-8)
    normed = jnp.clip(normed, -5.0, 5.0)
    do_norm = count >= min_count
    return jnp.where(do_norm, normed, jnp.clip(obs, -10.0, 10.0)) 

RewardNormalizerState = tuple

def init_reward_normalizer() -> RewardNormalizerState:
    return (
        jnp.array(0.0), jnp.array(10.0), jnp.array(0),
        jnp.array(1e-4), jnp.array(100)
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

def normalize_reward(state, rewards: jnp.ndarray) -> jnp.ndarray: 
    mean, var, count, eps, min_count = state
    do_norm = count >= min_count
    normalized_or_raw = jnp.where(
        do_norm,
        (rewards - mean) / jnp.sqrt(var + eps + 1e-8),
        rewards
    )
    return jnp.clip(normalized_or_raw, -5.0, 5.0)


def sample_trajectories(
    task_env: Any, policy_params: Any, value_params: Any, policy_apply: Callable,
    value_apply: Callable, key: jnp.ndarray, obs_norm_state: Any,
    reward_norm_state: Any, num_steps: int = ROLLOUT_LENGTH, chunk_size: int = 10,
) -> Dict[str, jnp.ndarray]:
    state, timestep = task_env.reset(key)
    carry = (state, key)
    
    actual_chunk_size = chunk_size if num_steps >= chunk_size and chunk_size > 0 else num_steps
    actual_num_chunks = num_steps // actual_chunk_size if actual_chunk_size > 0 else 0
    remainder_steps = num_steps % actual_chunk_size if actual_chunk_size > 0 else 0
    if num_steps == 0 : actual_num_chunks = 0; remainder_steps = 0 


    def _chunk_step_body(carry_c, _):
        state_c, key_c = carry_c
        key_c, subkey_c = jax.random.split(key_c)
        obs_c = flatten_state(state_c)
        norm_obs_c = normalize_obs(obs_norm_state, obs_c)
        dist_c = policy_apply(policy_params, norm_obs_c)
        action_c = dist_c.sample(seed=subkey_c).reshape(-1)
        value_c = value_apply(value_params, norm_obs_c).squeeze()
        next_state_c, next_t_c = task_env.step(state_c, action_c)
        raw_reward_c = next_t_c.reward
        norm_reward_c = normalize_reward(reward_norm_state, raw_reward_c)
        done_c = next_t_c.last()
        sinr_viol_c = jnp.sum(task_env._compute_sinr(next_state_c) < task_env.min_sinr)
        qos_viol_c = jnp.sum(next_state_c.qos_metrics[:, 0] > task_env.max_latency)
        out_c = (norm_obs_c, action_c, norm_reward_c, raw_reward_c, value_c, done_c, sinr_viol_c, qos_viol_c)
        return (next_state_c, key_c), out_c

    all_outputs_list_components = [[] for _ in range(8)] 

    if actual_num_chunks > 0:
        def run_one_chunk(carry_runchunk, _):
            carry_runchunk, outs_runchunk = jax.lax.scan(_chunk_step_body, carry_runchunk, None, length=actual_chunk_size)
            return carry_runchunk, outs_runchunk
        carry, chunked_outputs_scan = jax.lax.scan(run_one_chunk, carry, None, length=actual_num_chunks)
        for i, component_chunk_data in enumerate(chunked_outputs_scan):
            all_outputs_list_components[i].append(component_chunk_data.reshape((-1,) + component_chunk_data.shape[2:]))


    if remainder_steps > 0:
        carry, remainder_outputs_scan = jax.lax.scan(_chunk_step_body, carry, None, length=remainder_steps)
        for i, component_rem_data in enumerate(remainder_outputs_scan):
            all_outputs_list_components[i].append(component_rem_data) 

    if num_steps == 0:
        s_init_flat = flatten_state(state)
        action_dim = task_env.action_spec().num_values.shape[0] if hasattr(task_env, 'action_spec') else 0
        return {
            "observations": jnp.empty((0, s_init_flat.shape[-1] if s_init_flat.ndim > 0 and s_init_flat.size > 0 else 0)),
            "actions": jnp.empty((0, action_dim)), "rewards": jnp.empty((0,)),
            "raw_rewards": jnp.empty((0,)), "values": jnp.empty((0,)),
            "dones": jnp.empty((0,), dtype=bool), "sinr_violations": jnp.empty((0,)),
            "qos_violations": jnp.empty((0,)), "final_value": jnp.array(0.0, dtype=jnp.float32),
        }

    final_traj_dict_keys = ["observations", "actions", "rewards", "raw_rewards", "values", "dones", "sinr_violations", "qos_violations"]
    final_traj_dict = {}
    for i, key_name in enumerate(final_traj_dict_keys):
        if all_outputs_list_components[i]: 
            final_traj_dict[key_name] = jnp.concatenate(all_outputs_list_components[i], axis=0)
        else:
            if key_name in ["observations"]: final_traj_dict[key_name] = jnp.empty((0, flatten_state(state).shape[-1]))
            elif key_name in ["actions"]: final_traj_dict[key_name] = jnp.empty((0, task_env.action_spec().num_values.shape[0]))
            else: final_traj_dict[key_name] = jnp.empty((0,))


    final_state_val = carry[0]
    final_obs = normalize_obs(obs_norm_state, flatten_state(final_state_val))
    final_value = value_apply(value_params, final_obs).squeeze()
    final_traj_dict["final_value"] = final_value
    
    return final_traj_dict


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
        # --- Masking approach for normalization ---
        mask_for_norm = (jnp.arange(actual_seq_len) < num_actual_steps).astype(jnp.float32)
        
        sum_adv = jnp.sum(adv_in_operand * mask_for_norm)
        # For variance calculation: E[X^2] - (E[X])^2
        sum_sq_adv = jnp.sum(jnp.square(adv_in_operand) * mask_for_norm)
        count_valid = jnp.maximum(jnp.sum(mask_for_norm), 1.0) 

        mean = sum_adv / count_valid
        
        var = jnp.maximum(0.0, (sum_sq_adv / count_valid) - jnp.square(mean))
        std = jnp.sqrt(var + 1e-6) 

        normalized_advantages = (adv_in_operand - mean) / std
        

        new_adv = normalized_advantages * mask_for_norm
        return new_adv
        

    def zeros_advantages_fn(adv_in_operand):
        return jnp.zeros_like(adv_in_operand)

    advantages = lax.cond(
        num_actual_steps > 0, 
        normalize_advantages_fn,
        zeros_advantages_fn,
        advantages 
    )
    
    advantages = jnp.nan_to_num(advantages) 
    return returns, advantages


def compute_inner_loss(policy_params, value_params, policy_apply, value_apply, traj): 
    obs = traj['observations']
    actions = traj['actions']
    returns = traj['returns']
    advantages = traj['advantages']
    dones = traj['dones']

    if obs.shape[0] == 0:
        zero_loss = jnp.array(0.0, dtype=jnp.float32)
        return zero_loss, (zero_loss, zero_loss)

    action_dist = policy_apply(policy_params, obs)
    entropy_terms = action_dist.entropy()
    log_prob_terms = action_dist.log_prob(actions)

    actual_seq_len = obs.shape[0]
    indices = jnp.arange(actual_seq_len)
    first_true_idx_or_len = jnp.min(jnp.where(dones, indices, actual_seq_len))
    num_valid_steps = jnp.minimum(first_true_idx_or_len + 1, actual_seq_len)
    valid_step_mask_float = (jnp.arange(actual_seq_len) < num_valid_steps).astype(jnp.float32)
    count_valid_steps = jnp.sum(valid_step_mask_float)
    safe_count_valid_steps = jnp.maximum(count_valid_steps, 1.0)

    if log_prob_terms.ndim > 1 and advantages.ndim == 1:
        advantages_reshaped = advantages.reshape((-1,) + (1,) * (log_prob_terms.ndim - 1))
    else:
        advantages_reshaped = advantages
    
    mask_for_terms = valid_step_mask_float
    if log_prob_terms.ndim > 1:
        mask_for_terms = jnp.expand_dims(valid_step_mask_float, axis=tuple(range(1, log_prob_terms.ndim)))

    policy_loss_val = -jnp.sum(log_prob_terms * advantages_reshaped * mask_for_terms) / safe_count_valid_steps \
                      - 0.01 * jnp.sum(entropy_terms * mask_for_terms) / safe_count_valid_steps
    policy_loss_val = jnp.clip(policy_loss_val, -100.0, 100.0)
    policy_loss_val = jnp.where(count_valid_steps > 0, policy_loss_val, 0.0)

    values_pred = value_apply(value_params, obs).squeeze()
    values_pred = jnp.nan_to_num(values_pred, nan=0.0)
    value_loss_val = jnp.sum(jnp.square(returns - values_pred) * valid_step_mask_float) / safe_count_valid_steps
    value_loss_val = jnp.clip(value_loss_val, 0.0, 100.0)
    value_loss_val = jnp.where(count_valid_steps > 0, value_loss_val, 0.0)

    combined_loss = policy_loss_val + 0.3 * value_loss_val
    return combined_loss, (policy_loss_val, value_loss_val)


def inner_adaptation(policy_params, value_params, policy_apply, value_apply, traj, inner_lr, inner_steps):
    def adaptation_step(carry, _):
        curr_policy_params, curr_value_params = carry
        grad_fn = jax.value_and_grad(
            lambda p_tuple: compute_inner_loss(
                p_tuple[0], p_tuple[1], policy_apply, value_apply, traj
            )[0], argnums=0, has_aux=False
        )
        inner_loss, grads_tuple = grad_fn((curr_policy_params, curr_value_params))
        policy_grads, value_grads = grads_tuple
        policy_grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), policy_grads)
        value_grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), value_grads)
        new_policy_params = jax.tree_map(lambda p, g: p - inner_lr * g, curr_policy_params, policy_grads)
        new_value_params = jax.tree_map(lambda p, g: p - inner_lr * g, curr_value_params, value_grads)
        return (new_policy_params, new_value_params), inner_loss

    init_params_tuple = (policy_params, value_params)
    (adapted_policy_params, adapted_value_params), _ = jax.lax.scan(
        adaptation_step, init_params_tuple, None, length=inner_steps
    )
    return adapted_policy_params, adapted_value_params


def prepare_trajectory(traj, value_params, value_apply):
    if traj['observations'].shape[0] == 0:
        return {**traj, 'returns': jnp.array([]), 'advantages': jnp.array([])}
    returns, advantages = compute_gae(
        traj['rewards'], traj['values'], traj['dones'], traj['final_value']
    )
    return {**traj, 'returns': returns, 'advantages': advantages}


def compute_meta_objective(init_params, train_traj, test_traj,
                          policy_apply, value_apply, inner_lr, inner_steps):
    init_policy_params, init_value_params = init_params
    adapted_policy_params, adapted_value_params = inner_adaptation(
        init_policy_params, init_value_params,
        policy_apply, value_apply,
        train_traj, inner_lr, inner_steps
    )
    test_loss, (policy_loss, value_loss) = compute_inner_loss(
        adapted_policy_params, adapted_value_params,
        policy_apply, value_apply, test_traj
    )
    return test_loss, (policy_loss, value_loss)


def sample_task(env: DynamicSpectrumEnv, key: chex.PRNGKey) -> DynamicSpectrumEnv: 
    key_interf, key_fading = jax.random.split(key)
    
    
    variation_interference = jax.random.uniform(key_interf, (),  minval=0.8, maxval=1.2)
    new_max_external_interference_mW = env.max_external_interference_mW * variation_interference

    fading_variation = jax.random.uniform(key_fading, (), minval=0.8, maxval=1.2)
    new_fading_coherence = env.fading_coherence * fading_variation

    new_env = DynamicSpectrumEnv(
        num_bs=env.num_bs,
        num_users=env.num_users,
        num_bands=env.num_bands,
        max_steps=env.max_steps,
        max_latency=env.max_latency,
        max_power_dbm=env.max_power_dbm, 
        num_power_levels=env.num_power_levels,
        power_levels_dbm=env.power_levels_dbm, 
        fading_coherence=new_fading_coherence,
        max_external_interference_mW=new_max_external_interference_mW,
        min_sinr_db=env.min_sinr_db, 
        bandwidth_hz= getattr(env, 'bandwidth_hz', BANDWIDTH_HZ), 
        noise_figure_db= getattr(env, 'noise_figure_db', NOISE_FIGURE_DB)
    )
    return new_env


def compute_meta_objective_for_task(
    params: Tuple[Any, Any], env: Any, policy_apply: Any, value_apply: Any,
    task_key: jnp.ndarray, inner_lr: float, inner_steps: int,
    obs_norm_state: ObsNormalizerState, reward_norm_state: RewardNormalizerState,
    rollout_len_static: int # Made explicit
) -> Tuple[jnp.ndarray, ObsNormalizerState, RewardNormalizerState, Dict[str, Any]]:
    task_sample_key, train_traj_key, test_traj_key = jax.random.split(task_key, 3)
    policy_params, value_params = params
    task_env_instance = sample_task(env, task_sample_key)

    train_traj_raw = sample_trajectories(
        task_env_instance, policy_params, value_params, policy_apply, value_apply, train_traj_key,
        obs_norm_state, reward_norm_state, num_steps=rollout_len_static
    )
    train_traj_prepared = prepare_trajectory(train_traj_raw, value_params, value_apply)
    
    adapted_policy_params, adapted_value_params = inner_adaptation(
        policy_params, value_params, policy_apply, value_apply,
        train_traj_prepared, inner_lr, inner_steps
    )
    test_traj_raw = sample_trajectories(
        task_env_instance, adapted_policy_params, adapted_value_params, policy_apply, value_apply, test_traj_key,
        obs_norm_state, reward_norm_state, num_steps=rollout_len_static
    )
    test_traj_prepared = prepare_trajectory(test_traj_raw, adapted_value_params, value_apply)
    
    meta_loss_this_task, (policy_loss_on_test, value_loss_on_test) = compute_meta_objective(
        params, train_traj_prepared, test_traj_prepared,
        policy_apply, value_apply, inner_lr, inner_steps
    )

    obs_to_normalize = []
    if train_traj_raw['observations'].shape[0] > 0: obs_to_normalize.append(train_traj_raw['observations'])
    if test_traj_raw['observations'].shape[0] > 0: obs_to_normalize.append(test_traj_raw['observations'])
    rewards_to_normalize = []
    if 'raw_rewards' in train_traj_raw and train_traj_raw['raw_rewards'].shape[0] > 0:
        rewards_to_normalize.append(train_traj_raw['raw_rewards'])
    if 'raw_rewards' in test_traj_raw and test_traj_raw['raw_rewards'].shape[0] > 0:
        rewards_to_normalize.append(test_traj_raw['raw_rewards'])

    current_task_obs_norm_state = obs_norm_state
    if obs_to_normalize:
        all_obs_this_task = jnp.concatenate(obs_to_normalize, axis=0)
        current_task_obs_norm_state = update_obs_normalizer(obs_norm_state, all_obs_this_task)
    current_task_reward_norm_state = reward_norm_state
    if rewards_to_normalize:
        all_rewards_this_task = jnp.concatenate(rewards_to_normalize, axis=0)
        current_task_reward_norm_state = update_reward_normalizer(reward_norm_state, all_rewards_this_task)
    
    aux_info = {"train_rewards_mean": jnp.mean(train_traj_prepared.get('rewards', jnp.array(0.0))),
                "test_rewards_mean": jnp.mean(test_traj_prepared.get('rewards', jnp.array(0.0)))}
    return meta_loss_this_task, current_task_obs_norm_state, current_task_reward_norm_state, aux_info


def adapt_to_task(
    meta_params: Tuple[Any, Any], env_instance: Any, policy_apply: Callable, value_apply: Callable,
    adaptation_key: jnp.ndarray, inner_lr: float, num_adaptation_steps: int,
    obs_norm_state: ObsNormalizerState, reward_norm_state: RewardNormalizerState,
    rollout_length_adaptation: int = ROLLOUT_LENGTH
) -> Tuple[Any, Any]:
    meta_policy_params, meta_value_params = meta_params
    if num_adaptation_steps == 0: return meta_params

    def adaptation_scan_body(carry_params_tuple, key_for_step):
        current_policy_p, current_value_p = carry_params_tuple
        support_traj_raw = sample_trajectories(
            task_env=env_instance, policy_params=current_policy_p, value_params=current_value_p,
            policy_apply=policy_apply, value_apply=value_apply, key=key_for_step,
            obs_norm_state=obs_norm_state, reward_norm_state=reward_norm_state,
            num_steps=rollout_length_adaptation
        )
        if support_traj_raw['observations'].shape[0] == 0:
            return (current_policy_p, current_value_p), jnp.array(0.0)
        support_traj_prepared = prepare_trajectory(support_traj_raw, current_value_p, value_apply)
        grad_fn = jax.value_and_grad(
            lambda p_tuple: compute_inner_loss(
                p_tuple[0], p_tuple[1], policy_apply, value_apply, support_traj_prepared
            )[0], argnums=0, has_aux=False
        )
        adaptation_loss, (policy_g, value_g) = grad_fn((current_policy_p, current_value_p))
        adapted_policy_p = jax.tree_map(lambda p, g: p - inner_lr * g, current_policy_p, policy_g)
        adapted_value_p = jax.tree_map(lambda p, g: p - inner_lr * g, current_value_p, value_g)
        return (adapted_policy_p, adapted_value_p), adaptation_loss

    adaptation_step_keys = jax.random.split(adaptation_key, num_adaptation_steps)
    initial_adapt_params = (meta_policy_params, meta_value_params)
    (final_adapted_policy_params, final_adapted_value_params), _ = jax.lax.scan(
        adaptation_scan_body, initial_adapt_params, adaptation_step_keys
    )
    return (final_adapted_policy_params, final_adapted_value_params)


def train_maml(
    env: Any, policy_params: Any, value_params: Any, policy_apply: Callable, value_apply: Callable,
    num_tasks: int, inner_lr: float, inner_steps: int, meta_lr: float, num_iterations: int,
    dim: int, key: jnp.ndarray, eval_interval: int = 10, num_eval_tasks: int = 5,
    wandb_project: str = "maml-training", wandb_name: str = None, use_wandb: bool = True,
    rollout_len_config: int = ROLLOUT_LENGTH
) -> Tuple[Tuple[Any,Any], Dict[str, list]]:

    initial_meta_params = (policy_params, value_params)
    lr_schedule = optax.exponential_decay(init_value=meta_lr, transition_steps=1000, decay_rate=0.95)
    optimizer = optax.chain(optax.clip(1.0), optax.adam(lr_schedule))

    if use_wandb:
        wandb.init(project=wandb_project, name=wandb_name, config={
            "num_tasks_per_meta_batch": num_tasks, "inner_lr": inner_lr, "inner_steps": inner_steps,
            "meta_lr": meta_lr, "num_iterations": num_iterations, "obs_dim": dim,
            "eval_interval": eval_interval, "num_eval_tasks": num_eval_tasks,
            "rollout_len": rollout_len_config
        })

    key, meta_iteration_keys_key = jax.random.split(key)
    meta_iteration_keys = jax.random.split(meta_iteration_keys_key, num_iterations)
    
   
    @partial(jax.jit, static_argnames=("num_tasks_static", "policy_apply_static", "value_apply_static", 
                                       "inner_lr_static", "inner_steps_static", "rollout_len_static_arg", "env_static"))
    def meta_iteration_update_step(
        carry_main_loop, key_for_meta_iter,
        num_tasks_static, policy_apply_static, value_apply_static, inner_lr_static, 
        inner_steps_static, rollout_len_static_arg, env_static 
    ):
        (current_meta_p, current_opt_s, current_obs_norm_s, current_reward_norm_s) = carry_main_loop
        task_keys_for_this_meta_batch = jax.random.split(key_for_meta_iter, num_tasks_static)
        zero_grads = jax.tree_map(jnp.zeros_like, current_meta_p)

        def process_one_task_in_meta_batch(carry_task_scan, task_key_for_grad):
            acc_grads_task, obs_norm_s_task, reward_norm_s_task = carry_task_scan
            
    
            def grad_target_fn(p_for_grad):
                # compute_meta_objective_for_task returns: loss, new_obs_norm, new_reward_norm, aux
                loss_val, _, _, _ = compute_meta_objective_for_task(
                    p_for_grad, env_static, policy_apply_static, value_apply_static,
                    task_key_for_grad, inner_lr_static, inner_steps_static,
                    obs_norm_s_task, reward_norm_s_task,
                    rollout_len_static=rollout_len_static_arg
                )
                return loss_val


            loss_this_task, updated_obs_norm_s_after_task, updated_reward_norm_s_after_task, aux_info_this_task = \
                compute_meta_objective_for_task(
                    current_meta_p, env_static, policy_apply_static, value_apply_static,
                    task_key_for_grad, inner_lr_static, inner_steps_static,
                    obs_norm_s_task, reward_norm_s_task,
                    rollout_len_static=rollout_len_static_arg
                )
            grads_this_task = jax.grad(grad_target_fn)(current_meta_p)


            new_acc_grads_task = jax.tree_map(lambda x, y: x + y, acc_grads_task, grads_this_task)
            return (new_acc_grads_task, updated_obs_norm_s_after_task, updated_reward_norm_s_after_task), \
                   (loss_this_task, aux_info_this_task) 

        initial_task_scan_carry = (zero_grads, current_obs_norm_s, current_reward_norm_s)
        (final_acc_grads_meta_batch, final_obs_norm_s_meta_batch, final_reward_norm_s_meta_batch), \
        (per_task_losses_meta_batch, _) = lax.scan( 
            process_one_task_in_meta_batch, initial_task_scan_carry, task_keys_for_this_meta_batch
        )

        avg_grads_meta_batch = jax.tree_map(lambda g: g / num_tasks_static, final_acc_grads_meta_batch)
        avg_loss_meta_batch = jnp.mean(per_task_losses_meta_batch)
        updates, new_opt_s = optimizer.update(avg_grads_meta_batch, current_opt_s, current_meta_p)
        new_meta_p = optax.apply_updates(current_meta_p, updates)
        next_carry_main_loop = (new_meta_p, new_opt_s, final_obs_norm_s_meta_batch, final_reward_norm_s_meta_batch)
        metrics_for_this_iteration = {"meta_loss": avg_loss_meta_batch, "grad_norm": optax.global_norm(avg_grads_meta_batch)}
        return next_carry_main_loop, metrics_for_this_iteration

    # Initialize states for the Python loop over iterations
    current_params_loop = initial_meta_params
    current_obs_norm_state_loop = init_obs_normalizer(dim)
    current_reward_norm_state_loop = init_reward_normalizer()
    current_opt_state_loop = optimizer.init(initial_meta_params)

    # History lists
    meta_losses_log = []
    avg_pre_rewards_log, avg_post_rewards_log, avg_reward_improvements_log = [], [], []
    avg_pre_sinrs_log, avg_post_sinrs_log, avg_sinr_improvements_log = [], [], []
    avg_pre_qoss_log, avg_post_qoss_log, avg_qos_improvements_log = [], [], []

    for iter_idx in range(num_iterations):
         #JIT cache periodically
        if iter_idx > 0 and iter_idx % 20 == 0:
            print(f"DEBUG: Clearing JAX JIT caches at iteration {iter_idx}")
            jax.clear_caches()
        

        
        (next_params_loop, next_opt_state_loop, next_obs_norm_loop, next_reward_norm_loop), metrics_iter = \
            meta_iteration_update_step(
                (current_params_loop, current_opt_state_loop, current_obs_norm_state_loop, current_reward_norm_state_loop),
                meta_iteration_keys[iter_idx],
                num_tasks, policy_apply, value_apply, inner_lr, inner_steps, rollout_len_config, env
            )
        
       
        current_params_loop = next_params_loop
        current_opt_state_loop = next_opt_state_loop
        current_obs_norm_state_loop = next_obs_norm_loop
        current_reward_norm_state_loop = next_reward_norm_loop
        
        current_iter_loss = float(metrics_iter["meta_loss"])
        meta_losses_log.append(current_iter_loss)

        log_payload = {"iteration": iter_idx, "meta_loss": current_iter_loss}
        
        if iter_idx % eval_interval == 0 or iter_idx == num_iterations - 1:
            params_for_eval = current_params_loop
            obs_norm_for_eval = current_obs_norm_state_loop
            reward_norm_for_eval = current_reward_norm_state_loop
            
            
            key_eval_iter, eval_master_key_loop = jax.random.split(meta_iteration_keys[iter_idx]) 
            eval_task_keys_loop = jax.random.split(eval_master_key_loop, num_eval_tasks)
            
            pre_rewards_b, post_rewards_b = [], []
            pre_sinrs_b, post_sinrs_b = [], []
            pre_qoss_b, post_qoss_b = [], []

            for eval_tk_single in eval_task_keys_loop:
                eval_task_sample_key, eval_traj_master_key = jax.random.split(eval_tk_single)
                eval_task_env_loop = sample_task(env, eval_task_sample_key)
                
                pre_key, adapt_key_loop, post_key = jax.random.split(eval_traj_master_key, 3)

                pre_traj_ev = sample_trajectories(eval_task_env_loop, params_for_eval[0], params_for_eval[1], policy_apply, value_apply, pre_key, obs_norm_for_eval, reward_norm_for_eval, num_steps=rollout_len_config)
                if pre_traj_ev['observations'].shape[0] > 0:
                    pre_rewards_b.append(jnp.mean(pre_traj_ev["rewards"]))
                    pre_sinrs_b.append(jnp.mean(pre_traj_ev["sinr_violations"]))
                    pre_qoss_b.append(jnp.mean(pre_traj_ev["qos_violations"]))
                
                adapted_p_ev = adapt_to_task(params_for_eval, eval_task_env_loop, policy_apply, value_apply, adapt_key_loop, inner_lr, inner_steps, obs_norm_for_eval, reward_norm_for_eval, rollout_length_adaptation=rollout_len_config)
                post_traj_ev = sample_trajectories(eval_task_env_loop, adapted_p_ev[0], adapted_p_ev[1], policy_apply, value_apply, post_key, obs_norm_for_eval, reward_norm_for_eval, num_steps=rollout_len_config)
                if post_traj_ev['observations'].shape[0] > 0:
                    post_rewards_b.append(jnp.mean(post_traj_ev["rewards"]))
                    post_sinrs_b.append(jnp.mean(post_traj_ev["sinr_violations"]))
                    post_qoss_b.append(jnp.mean(post_traj_ev["qos_violations"]))

            if pre_rewards_b: 
                apr = jnp.mean(jnp.array(pre_rewards_b))
                apsr = jnp.mean(jnp.array(post_rewards_b) if post_rewards_b else jnp.array(pre_rewards_b)) 
                asi = apsr - apr
                avg_pre_rewards_log.append(float(apr)); avg_post_rewards_log.append(float(apsr)); avg_reward_improvements_log.append(float(asi))
                log_payload.update({"avg_pre_reward": float(apr), "avg_post_reward": float(apsr), "avg_reward_improvement": float(asi)})
            if pre_sinrs_b:
                aps = jnp.mean(jnp.array(pre_sinrs_b))
                aops = jnp.mean(jnp.array(post_sinrs_b) if post_sinrs_b else jnp.array(pre_sinrs_b))
                assi = aps - aops
                avg_pre_sinrs_log.append(float(aps)); avg_post_sinrs_log.append(float(aops)); avg_sinr_improvements_log.append(float(assi))
                log_payload.update({"avg_pre_sinr_violation": float(aps), "avg_post_sinr_violation": float(aops), "avg_sinr_improvement": float(assi)})
            if pre_qoss_b:
                apq = jnp.mean(jnp.array(pre_qoss_b))
                apoq = jnp.mean(jnp.array(post_qoss_b) if post_qoss_b else jnp.array(pre_qoss_b))
                asqi = apq - apoq
                avg_pre_qoss_log.append(float(apq)); avg_post_qoss_log.append(float(apoq)); avg_qos_improvements_log.append(float(asqi))
                log_payload.update({"avg_pre_qos_violation": float(apq), "avg_post_qos_violation": float(apoq), "avg_qos_improvement": float(asqi)})

            print(f"[Iter {iter_idx}] meta_loss={log_payload.get('meta_loss', 0.0):.3f} "
                  f"pre_r={log_payload.get('avg_pre_reward', 0.0):.3f} post_r={log_payload.get('avg_post_reward', 0.0):.3f}")
            if use_wandb: wandb.log(log_payload, step=iter_idx)

    trained_params = current_params_loop

    if use_wandb: wandb.finish()
    history = {
        "meta_losses": meta_losses_log,
        "avg_pre_rewards": avg_pre_rewards_log, "avg_post_rewards": avg_post_rewards_log, "avg_reward_improvements": avg_reward_improvements_log,
        "avg_pre_sinrs": avg_pre_sinrs_log, "avg_post_sinrs": avg_post_sinrs_log, "avg_sinr_improvements": avg_sinr_improvements_log,
        "avg_pre_qoss": avg_pre_qoss_log, "avg_post_qoss": avg_post_qoss_log, "avg_qos_improvements": avg_qos_improvements_log,
    }
    return trained_params, history
