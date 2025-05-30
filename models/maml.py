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


# Constants
NUM_BS = 3
NUM_BANDS = 4
NUM_USERS = 5
NUM_POWER_LEVELS = 5

# Hyperparameters
META_LR = 1e-3
INNER_LR = 0.1
META_BATCH_SIZE = 4 # Corresponds to num_tasks in train_maml
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

ObsNormalizerState = tuple  

def init_obs_normalizer(obs_dim: int) -> ObsNormalizerState:
    return (
        jnp.zeros(obs_dim), 
        jnp.ones(obs_dim) * 10.0 ,   
        jnp.array(0),        
        jnp.array(1e-4),     
        jnp.array(100)       
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


RewardNormalizerState = tuple  

def init_reward_normalizer() -> RewardNormalizerState:
    return (
        jnp.array(0.0),  
        jnp.array(10.0),  
        jnp.array(0),    
        jnp.array(1e-4), 
        jnp.array(100)   
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


def normalize_obs(state, obs: jnp.ndarray) -> jnp.ndarray:
    mean, var, count, eps, min_count = state
    normed = (obs - mean) / jnp.sqrt(var + eps)
    normed = jnp.clip(normed, -5.0, 5.0)
    do_norm = count >= min_count
    return jnp.where(do_norm, normed, obs)


def normalize_reward(state, rewards: jnp.ndarray) -> jnp.ndarray:
    mean, var, count, eps, min_count = state
    normed = (rewards - mean) / jnp.sqrt(var + eps)
    normed = jnp.clip(normed, -3.0, 3.0)
    do_norm = count >= min_count
    return jnp.where(do_norm, normed, rewards)


def sample_trajectories(
    task_env: Any,
    policy_params: Any,
    value_params: Any,
    policy_apply: Callable,
    value_apply: Callable,
    key: jnp.ndarray,
    obs_norm_state: Any,
    reward_norm_state: Any,
    num_steps: int = ROLLOUT_LENGTH, 
    chunk_size: int = 10,
) -> Dict[str, jnp.ndarray]:
    state, timestep = task_env.reset(key)
    carry = (state, key)

    def chunk_step(carry, _):
        state, key = carry
        key, subkey = jax.random.split(key)
        obs = flatten_state(state)
        norm_obs = normalize_obs(obs_norm_state, obs)

        dist = policy_apply(policy_params, norm_obs)
        action = dist.sample(seed=subkey).reshape(-1)
        value = value_apply(value_params, norm_obs).squeeze()

        next_state, next_t = task_env.step(state, action)
        raw_reward = next_t.reward
        norm_reward = normalize_reward(reward_norm_state, raw_reward)
        done = next_t.last()

        sinr_viol = jnp.sum(task_env._compute_sinr(next_state) < task_env.min_sinr)
        qos_viol  = jnp.sum(next_state.qos_metrics[:, 0] > task_env.max_latency)

        out = (norm_obs, action, norm_reward, value, done, sinr_viol, qos_viol)
        return (next_state, key), out

    if num_steps % chunk_size != 0:
        pass
    num_chunks = num_steps // chunk_size
    
    if num_chunks == 0:
        if num_steps == 0:
            dummy_obs_shape = flatten_state(state).shape
            dummy_action_shape = (env.num_bs * env.num_bands,) 
            return {
                "observations": jnp.empty((0, *dummy_obs_shape)),
                "actions": jnp.empty((0, *dummy_action_shape)),
                "rewards": jnp.empty((0,)), "values": jnp.empty((0,)),
                "dones": jnp.empty((0,), dtype=bool),
                "sinr_violations": jnp.empty((0,)), "qos_violations": jnp.empty((0,)),
                "final_value": jnp.array(0.0),
            }
        carry_final, chunk_outputs_single = jax.lax.scan(chunk_step, carry, None, length=num_steps)
        all_o, all_a, all_r, all_v, all_d, all_s, all_q = jax.tree_map(lambda x: x[jnp.newaxis, ...], chunk_outputs_single) 
        T = num_steps
    else:
        def run_chunked(carry_chunk, _):
            carry_chunk, outs_chunk = jax.lax.scan(chunk_step, carry_chunk, None, length=chunk_size)
            return carry_chunk, outs_chunk
        carry_final, chunk_outputs = jax.lax.scan(run_chunked, carry, None, length=num_chunks)
        all_o, all_a, all_r, all_v, all_d, all_s, all_q = chunk_outputs
        T = num_chunks * chunk_size


    observations    = all_o.reshape((T, -1))
    actions         = all_a.reshape((T, all_a.shape[-1]))
    rewards         = all_r.reshape((T,))
    values          = all_v.reshape((T,))
    dones           = all_d.reshape((T,))
    sinr_violations = all_s.reshape((T,))
    qos_violations  = all_q.reshape((T,))

    final_state_val = carry_final[0]
    final_obs = normalize_obs(obs_norm_state, flatten_state(final_state_val))
    final_value = value_apply(value_params, final_obs).squeeze()

    return {
        "observations":    observations,
        "actions":         actions,
        "rewards":         rewards,
        "values":          values,
        "dones":           dones,
        "sinr_violations": sinr_violations,
        "qos_violations":  qos_violations,
        "final_value":     final_value,
    }


def compute_gae(rewards, values, dones, final_value, gamma=0.99, lambda_=0.95):
    episode_length = len(rewards)
    if episode_length == 0: # Check if rewards is empty
        return jnp.array([]), jnp.array([])

    advantages = jnp.zeros_like(rewards)

    next_value = final_value
    next_advantage = 0.0

    for t in reversed(range(episode_length)):
        mask = 1.0 - dones[t].astype(jnp.float32)
        td_error = rewards[t] + gamma * next_value * mask - values[t]
        
        # Correct GAE calculation for advantage
        advantages_t = td_error + gamma * lambda_ * next_advantage * mask
        advantages = advantages.at[t].set(jnp.nan_to_num(advantages_t)) 
        
        next_value = values[t]
        next_advantage = advantages[t] 

    returns = advantages + values
    
    
    adv_mean = jnp.mean(advantages)
    adv_std = jnp.std(advantages) + 1e-6 # Increased epsilon for stability
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

    if log_probs.ndim > 1 and advantages.ndim == 1:
        advantages_reshaped = advantages.reshape((-1,) + (1,) * (log_probs.ndim - 1))
    else:
        advantages_reshaped = advantages

    policy_loss = -jnp.mean(log_probs * advantages_reshaped) - 0.01 * jnp.mean(entropy)
    policy_loss = jnp.clip(policy_loss, -100.0, 100.0)


    values_pred = value_apply(value_params, obs).squeeze() 
    values_pred = jnp.nan_to_num(values_pred, nan=0.0)
    value_loss = jnp.mean(jnp.square(returns - values_pred))
    value_loss = jnp.clip(value_loss, 0.0, 100.0)


    combined_loss = policy_loss + 0.3 * value_loss

    return combined_loss, (policy_loss, value_loss)


def inner_adaptation(policy_params, value_params, policy_apply, value_apply, traj, inner_lr, inner_steps):

    def adaptation_step(carry, _):
        curr_policy_params, curr_value_params = carry
        


        grad_fn = jax.value_and_grad(
            lambda p_tuple: compute_inner_loss( 
                p_tuple[0], p_tuple[1], policy_apply, value_apply, traj 
            )[0], 
            argnums=0, 
            has_aux=False
        )
        inner_loss, grads_tuple = grad_fn((curr_policy_params, curr_value_params))
        policy_grads, value_grads = grads_tuple 

        policy_grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), policy_grads)
        value_grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), value_grads)

        new_policy_params = jax.tree_map(
            lambda p, g: p - inner_lr * g, curr_policy_params, policy_grads
        )
        new_value_params = jax.tree_map(
            lambda p, g: p - inner_lr * g, curr_value_params, value_grads
        )

        return (new_policy_params, new_value_params), inner_loss 

    init_params_tuple = (policy_params, value_params)
    (adapted_policy_params, adapted_value_params), inner_losses = jax.lax.scan(
        adaptation_step,
        init_params_tuple,
        None,
        length=inner_steps
    )

    return adapted_policy_params, adapted_value_params


def prepare_trajectory(traj, value_params, value_apply):
    if traj['observations'].shape[0] == 0:
        return {**traj, 'returns': jnp.array([]), 'advantages': jnp.array([])}

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


def sample_task(env, key):
    key_interference, key_fading = jax.random.split(key)

    variation_interference = jax.random.uniform(key_interference, (),  minval=0.9, maxval=1.1)
    new_max_interference = env.max_interference * variation_interference

    fading_var = jax.random.uniform(key_fading, (), minval=0.9, maxval=1.1)
    new_fading = env.fading_coherence * fading_var

    new_env = DynamicSpectrumEnv(
        num_bs=env.num_bs,
        num_users=env.num_users,
        num_bands=env.num_bands,
        max_steps=env.max_steps,
        max_latency=env.max_latency,
        max_power=env.max_power,
        num_power_levels=env.num_power_levels,
        power_levels=env.power_levels,
        fading_coherence=new_fading,
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
) -> Tuple[jnp.ndarray, ObsNormalizerState, RewardNormalizerState, Dict[str, Any], Dict[str, Any]]:

    task_sample_key, train_traj_key, test_traj_key = jax.random.split(task_key, 3)
    policy_params, value_params = params 
    task_env_instance = sample_task(env, task_sample_key)
    train_traj_raw = sample_trajectories(
        task_env_instance, policy_params, value_params, policy_apply, value_apply, train_traj_key,
        obs_norm_state, reward_norm_state, num_steps=ROLLOUT_LENGTH 
    )
    train_traj_prepared = prepare_trajectory(train_traj_raw, value_params, value_apply)
    adapted_policy_params, adapted_value_params = inner_adaptation(
        policy_params, value_params, policy_apply, value_apply,
        train_traj_prepared, 
        inner_lr, inner_steps
    )
    test_traj_raw = sample_trajectories(
        task_env_instance, adapted_policy_params, adapted_value_params, policy_apply, value_apply, test_traj_key,
        obs_norm_state, reward_norm_state, num_steps=ROLLOUT_LENGTH 
    )
    test_traj_prepared = prepare_trajectory(test_traj_raw, adapted_value_params, value_apply)
    meta_loss_this_task, (policy_loss_on_test, value_loss_on_test) = compute_inner_loss(
        adapted_policy_params, adapted_value_params,
        policy_apply, value_apply, test_traj_prepared
    )
    obs_to_normalize = []
    if train_traj_raw['observations'].shape[0] > 0:
        obs_to_normalize.append(train_traj_raw['observations'])
    if test_traj_raw['observations'].shape[0] > 0:
        obs_to_normalize.append(test_traj_raw['observations'])

    rewards_to_normalize = []
    if train_traj_raw['rewards'].shape[0] > 0: 
        rewards_to_normalize.append(train_traj_raw['rewards']) 
    if test_traj_raw['rewards'].shape[0] > 0:
        rewards_to_normalize.append(test_traj_raw['rewards'])

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
    meta_params: Tuple[Any, Any],       
    env_instance: Any,                  
    policy_apply: Callable,
    value_apply: Callable,
    adaptation_key: jnp.ndarray,        
    inner_lr: float,                    
    num_adaptation_steps: int,          
    obs_norm_state: ObsNormalizerState, 
    reward_norm_state: RewardNormalizerState, 
    rollout_length_adaptation: int = ROLLOUT_LENGTH 
) -> Tuple[Any, Any]: 
    """
    Adapts the meta-learned parameters to a specific task environment for evaluation.
    """
    meta_policy_params, meta_value_params = meta_params

    if num_adaptation_steps == 0:
        return meta_params 

    def adaptation_scan_body(carry_params_tuple, key_for_step):
        current_policy_p, current_value_p = carry_params_tuple
        
        
        support_traj_raw = sample_trajectories(
            task_env=env_instance,
            policy_params=current_policy_p, 
            value_params=current_value_p,
            policy_apply=policy_apply,
            value_apply=value_apply,
            key=key_for_step,
            obs_norm_state=obs_norm_state,
            reward_norm_state=reward_norm_state,
            num_steps=rollout_length_adaptation
        )

       
        if support_traj_raw['observations'].shape[0] == 0:
            return (current_policy_p, current_value_p), None 

        
        support_traj_prepared = prepare_trajectory(support_traj_raw, current_value_p, value_apply)

        grad_fn = jax.value_and_grad(
            lambda p_tuple: compute_inner_loss(
                p_tuple[0], p_tuple[1], policy_apply, value_apply, support_traj_prepared
            )[0], 
            argnums=0, has_aux=False
        )
        adaptation_loss, (policy_g, value_g) = grad_fn((current_policy_p, current_value_p))

        # Apply gradient update (SGD)
        adapted_policy_p = jax.tree_map(
            lambda p, g: p - inner_lr * g, current_policy_p, policy_g
        )
        adapted_value_p = jax.tree_map(
            lambda p, g: p - inner_lr * g, current_value_p, value_g
        )
        return (adapted_policy_p, adapted_value_p), adaptation_loss

    adaptation_step_keys = jax.random.split(adaptation_key, num_adaptation_steps)
    
    initial_adapt_params = (meta_policy_params, meta_value_params)
    
    (final_adapted_policy_params, final_adapted_value_params), adaptation_losses = jax.lax.scan(
        adaptation_scan_body,
        initial_adapt_params,
        adaptation_step_keys
    )
    
    return (final_adapted_policy_params, final_adapted_value_params)

def train_maml(
    env: Any, 
    policy_params: Any, 
    value_params: Any,  
    policy_apply: Callable,
    value_apply: Callable,
    num_tasks: int, 
    inner_lr: float,
    inner_steps: int,
    meta_lr: float,
    num_iterations: int, 
    dim: int, 
    key: jnp.ndarray, 
    eval_interval: int = 10,
    num_eval_tasks: int = 5, # Number of tasks for evaluation
    wandb_project: str = "maml-training",
    wandb_name: str = None,
    use_wandb: bool = True,
) -> Tuple[Tuple[Any,Any], Dict[str, list]]:

    initial_meta_params = (policy_params, value_params)

    lr_schedule = optax.exponential_decay(init_value=meta_lr, transition_steps=1000, decay_rate=0.95)
    optimizer = optax.chain(optax.clip(1.0), optax.adam(lr_schedule))

    if use_wandb:
        wandb.init(project=wandb_project, name=wandb_name, config={
            "num_tasks_per_meta_batch": num_tasks, "inner_lr": inner_lr, "inner_steps": inner_steps,
            "meta_lr": meta_lr, "num_iterations": num_iterations, "obs_dim": dim,
            "eval_interval": eval_interval, "num_eval_tasks": num_eval_tasks
        })

    key, meta_iteration_keys_key = jax.random.split(key)
    meta_iteration_keys = jax.random.split(meta_iteration_keys_key, num_iterations)

    @jax.jit
    def meta_iteration_update_step(carry_main_loop, key_for_meta_iter):
        (current_meta_p, current_opt_s, current_obs_norm_s, current_reward_norm_s) = carry_main_loop
        
        task_keys_for_this_meta_batch = jax.random.split(key_for_meta_iter, num_tasks)
        zero_grads = jax.tree_map(jnp.zeros_like, current_meta_p)

        # Function to process one task within the meta-batch
        def process_one_task_in_meta_batch(carry_task_scan, task_key_for_grad):
            acc_grads_task, obs_norm_s_task, reward_norm_s_task = carry_task_scan
            
            def task_loss_and_aux_fn(meta_p_for_grad):
                loss, obs_n_upd, reward_n_upd, aux = compute_meta_objective_for_task(
                    meta_p_for_grad, env, policy_apply, value_apply,
                    task_key_for_grad, inner_lr, inner_steps,
                    obs_norm_s_task, reward_norm_s_task
                )
                return loss, (obs_n_upd, reward_n_upd, aux)

            (loss_this_task, (obs_norm_updated_after_task, reward_norm_updated_after_task, aux_info_task)), grads_this_task = \
                jax.value_and_grad(task_loss_and_aux_fn, has_aux=True)(current_meta_p)

            new_acc_grads_task = jax.tree_map(lambda x, y: x + y, acc_grads_task, grads_this_task)
            
            return (new_acc_grads_task, obs_norm_updated_after_task, reward_norm_updated_after_task), \
                   (loss_this_task, aux_info_task)

        initial_task_scan_carry = (zero_grads, current_obs_norm_s, current_reward_norm_s)
        
        (final_acc_grads_meta_batch, final_obs_norm_s_meta_batch, final_reward_norm_s_meta_batch), \
        (per_task_losses_meta_batch, per_task_aux_info_meta_batch) = jax.lax.scan(
            process_one_task_in_meta_batch,
            initial_task_scan_carry,
            task_keys_for_this_meta_batch
        )

        avg_grads_meta_batch = jax.tree_map(lambda g: g / num_tasks, final_acc_grads_meta_batch)
        avg_loss_meta_batch = jnp.mean(per_task_losses_meta_batch)

        updates, new_opt_s = optimizer.update(avg_grads_meta_batch, current_opt_s, current_meta_p)
        new_meta_p = optax.apply_updates(current_meta_p, updates)
        
        next_carry_main_loop = (new_meta_p, new_opt_s, final_obs_norm_s_meta_batch, final_reward_norm_s_meta_batch)
        
        metrics_for_this_iteration = {
            "meta_loss": avg_loss_meta_batch,
            "grad_norm": optax.global_norm(avg_grads_meta_batch),
        }
           
        return next_carry_main_loop, metrics_for_this_iteration
    obs_norm_state_initial = init_obs_normalizer(dim)
    reward_norm_state_initial = init_reward_normalizer()
    opt_state_initial = optimizer.init(initial_meta_params)
    initial_main_loop_scan_carry = (initial_meta_params, opt_state_initial, obs_norm_state_initial, reward_norm_state_initial)

    (final_meta_p_tuple, _, final_obs_norm_s_overall, final_reward_norm_s_overall), iteration_metrics_all = jax.lax.scan(
        meta_iteration_update_step,
        initial_main_loop_scan_carry,
        meta_iteration_keys 
    )
    
    trained_params = final_meta_p_tuple 

    meta_losses_log = []
    avg_pre_rewards_log, avg_post_rewards_log, avg_reward_improvements_log = [], [], []
    avg_pre_sinrs_log, avg_post_sinrs_log, avg_sinr_improvements_log = [], [], []
    avg_pre_qoss_log, avg_post_qoss_log, avg_qos_improvements_log = [], [], []

    for iter_idx in range(num_iterations):
        current_iter_loss = float(iteration_metrics_all["meta_loss"][iter_idx])
        meta_losses_log.append(current_iter_loss)

        log_payload = {"iteration": iter_idx, "meta_loss": current_iter_loss}
        
        if iter_idx % eval_interval == 0:
            
            key, eval_master_key = jax.random.split(meta_iteration_keys[iter_idx]) 
            eval_task_keys = jax.random.split(eval_master_key, num_eval_tasks)

            pre_rewards_batch, post_rewards_batch = [], []
            pre_sinrs_batch, post_sinrs_batch = [], []
            pre_qoss_batch, post_qoss_batch = [], []

            params_for_eval = trained_params
            obs_norm_for_eval = final_obs_norm_s_overall
            reward_norm_for_eval = final_reward_norm_s_overall

            for eval_task_key_single in eval_task_keys:
                eval_task_env = sample_task(env, eval_task_key_single)
                key_s, pre_traj_key, adapt_key, post_traj_key = jax.random.split(eval_task_key_single, 4)

                pre_traj_eval = sample_trajectories(
                    eval_task_env, params_for_eval[0], params_for_eval[1],
                    policy_apply, value_apply, pre_traj_key,
                    obs_norm_for_eval, reward_norm_for_eval, num_steps=ROLLOUT_LENGTH
                )
                if pre_traj_eval['observations'].shape[0] > 0:
                    pre_rewards_batch.append(jnp.mean(pre_traj_eval["rewards"]))
                    pre_sinrs_batch.append(jnp.mean(pre_traj_eval["sinr_violations"]))
                    pre_qoss_batch.append(jnp.mean(pre_traj_eval["qos_violations"]))

                # Adapt parameters
                adapted_params_eval = adapt_to_task(
                    meta_params=params_for_eval,
                    env_instance=eval_task_env,
                    policy_apply=policy_apply,
                    value_apply=value_apply,
                    adaptation_key=adapt_key,
                    inner_lr=inner_lr, 
                    num_adaptation_steps=inner_steps, 
                    obs_norm_state=obs_norm_for_eval,
                    reward_norm_state=reward_norm_for_eval,
                    rollout_length_adaptation=ROLLOUT_LENGTH
                )

                # Post-adaptation evaluation
                post_traj_eval = sample_trajectories(
                    eval_task_env, adapted_params_eval[0], adapted_params_eval[1],
                    policy_apply, value_apply, post_traj_key,
                    obs_norm_for_eval, reward_norm_for_eval, num_steps=ROLLOUT_LENGTH
                )
                if post_traj_eval['observations'].shape[0] > 0:
                    post_rewards_batch.append(jnp.mean(post_traj_eval["rewards"]))
                    post_sinrs_batch.append(jnp.mean(post_traj_eval["sinr_violations"]))
                    post_qoss_batch.append(jnp.mean(post_traj_eval["qos_violations"]))
            
            if pre_rewards_batch: 
                apr = jnp.mean(jnp.array(pre_rewards_batch))
                apsr = jnp.mean(jnp.array(post_rewards_batch)) if post_rewards_batch else apr
                asi = apsr - apr
                avg_pre_rewards_log.append(float(apr)); avg_post_rewards_log.append(float(apsr)); avg_reward_improvements_log.append(float(asi))
                log_payload.update({"avg_pre_reward": float(apr), "avg_post_reward": float(apsr), "avg_reward_improvement": float(asi)})

            if pre_sinrs_batch:
                aps = jnp.mean(jnp.array(pre_sinrs_batch))
                aops = jnp.mean(jnp.array(post_sinrs_batch)) if post_sinrs_batch else aps
                assi = aps - aops 
                avg_pre_sinrs_log.append(float(aps)); avg_post_sinrs_log.append(float(aops)); avg_sinr_improvements_log.append(float(assi))
                log_payload.update({"avg_pre_sinr_violation": float(aps), "avg_post_sinr_violation": float(aops), "avg_sinr_improvement": float(assi)})

            if pre_qoss_batch:
                apq = jnp.mean(jnp.array(pre_qoss_batch))
                apoq = jnp.mean(jnp.array(post_qoss_batch)) if post_qoss_batch else apq
                asqi = apq - apoq 
                avg_pre_qoss_log.append(float(apq)); avg_post_qoss_log.append(float(apoq)); avg_qos_improvements_log.append(float(asqi))
                log_payload.update({"avg_pre_qos_violation": float(apq), "avg_post_qos_violation": float(apoq), "avg_qos_improvement": float(asqi)})
            
            print(f"[Iter {iter_idx}] meta_loss={current_iter_loss:.3f} "
                  f"pre_r={log_payload.get('avg_pre_reward', 0.0):.3f} post_r={log_payload.get('avg_post_reward', 0.0):.3f}")

        if use_wandb:
            wandb.log(log_payload, step=iter_idx)


    if use_wandb:
        wandb.finish()

    history = {
        "meta_losses": meta_losses_log,
        "avg_pre_rewards": avg_pre_rewards_log, "avg_post_rewards": avg_post_rewards_log, "avg_reward_improvements": avg_reward_improvements_log,
        "avg_pre_sinrs": avg_pre_sinrs_log, "avg_post_sinrs": avg_post_sinrs_log, "avg_sinr_improvements": avg_sinr_improvements_log,
        "avg_pre_qoss": avg_pre_qoss_log, "avg_post_qoss": avg_post_qoss_log, "avg_qos_improvements": avg_qos_improvements_log,
    }
    return trained_params, history
