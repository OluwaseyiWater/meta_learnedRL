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


ObsNormalizerState = tuple

def init_obs_normalizer(obs_dim: int) -> ObsNormalizerState:
    return (
        jnp.zeros(obs_dim),
        jnp.ones(obs_dim) * 10.0,
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


def normalize_obs(state, obs):
    mean, var, count, eps, min_count = state
    normed = (obs - mean) / jnp.sqrt(var + eps)
    normed = jnp.clip(normed, -3.0, 3.0)
    do_norm = count >= min_count
    return jnp.where(do_norm, normed, obs)

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
    
    def safe_squeeze_hidden_state_component(component):
        return jnp.squeeze(component, axis=0) if component.ndim == 2 and component.shape[0] == 1 else component

    state, timestep = task_env.reset(key)
    initial_obs = flatten_state(state)
    initial_norm_obs = normalize_obs(obs_norm_state, initial_obs)

    _, initial_hidden_state = policy_apply(policy_params, initial_norm_obs, None)

    def step_fn(carry, _): 
        env_state, current_hidden_s, current_norm_o, current_key = carry

        key_step, action_key_step = jax.random.split(current_key)
        
        
        action_dist, next_hidden_s = policy_apply(policy_params, current_norm_o, current_hidden_s)
        action_sample = action_dist.sample(seed=action_key_step) 
        log_prob_sample = action_dist.log_prob(action_sample)
        
        value_estimate = value_apply(value_params, current_norm_o).squeeze()

        action_flat_for_env = action_sample.reshape(-1) 
        next_env_state, next_env_timestep = task_env.step(env_state, action_flat_for_env)

        raw_reward = next_env_timestep.reward
        norm_reward_val = normalize_reward(reward_norm_state, raw_reward)

        sinr_vals = task_env._compute_sinr(next_env_state) 
        sinr_violation_count = jnp.sum(sinr_vals < task_env.min_sinr)
        qos_violation_count = jnp.sum(next_env_state.qos_metrics[:, 0] > task_env.max_latency)

        next_flat_obs = flatten_state(next_env_state)
        next_norm_o = normalize_obs(obs_norm_state, next_flat_obs)
        done_flag = next_env_timestep.last()

        new_carry = (next_env_state, next_hidden_s, next_norm_o, key_step)
        
        
        current_hidden_h_unbatched = current_hidden_s.hidden
        current_hidden_c_unbatched = current_hidden_s.cell

        outputs_step = (
            current_norm_o,        
            action_sample,         
            norm_reward_val,        
            value_estimate,         
            done_flag,              
            log_prob_sample,        
            current_hidden_h_unbatched, 
            current_hidden_c_unbatched, 
            sinr_violation_count,
            qos_violation_count,
            raw_reward              
        )
        return new_carry, outputs_step

    
    scan_initial_carry = (state, initial_hidden_state, initial_norm_obs, key)
    
    
    final_carry, scan_outputs = jax.lax.scan(step_fn, scan_initial_carry, xs=None, length=num_steps)

    
    (observations_all, actions_all, norm_rewards_all, values_all, dones_all, log_probs_all,
     hiddens_all, cells_all, sinr_violations_all, qos_violations_all, raw_rewards_all) = scan_outputs

    
    truncate_idx = num_steps
    if dones_all.any():
        truncate_idx = jnp.argmax(dones_all).item() + 1 

    observations_final = observations_all[:truncate_idx]
    actions_final = actions_all[:truncate_idx]
    rewards_final = norm_rewards_all[:truncate_idx] 
    values_final = values_all[:truncate_idx]
    dones_final = dones_all[:truncate_idx]
    log_probs_final = log_probs_all[:truncate_idx]
    hiddens_final = hiddens_all[:truncate_idx]
    cells_final = cells_all[:truncate_idx]
    sinr_violations_final = sinr_violations_all[:truncate_idx]
    qos_violations_final = qos_violations_all[:truncate_idx]
    raw_rewards_final = raw_rewards_all[:truncate_idx] 

    
    hidden_states_list = [hk.LSTMState(hidden=h, cell=c) for h, c in zip(hiddens_final, cells_final)]

    # Final value for GAE 
    final_norm_obs_for_gae = final_carry[2] 
    final_value_for_gae = value_apply(value_params, final_norm_obs_for_gae).squeeze()
    

    return {
        'observations':      observations_final,
        'actions':           actions_final,
        'rewards':           rewards_final, 
        'raw_rewards':       raw_rewards_final, 
        'values':            values_final,
        'dones':             dones_final,
        'log_probs':         log_probs_final, 
        'hidden_states':     hidden_states_list, 
        'final_value':       final_value_for_gae,
        'final_hidden_state': final_carry[1], 
        'sinr_violations':   sinr_violations_final,
        'qos_violations':    qos_violations_final,
    }


def compute_gae(rewards, values, dones, final_value, gamma=0.99, lambda_=0.95):
    episode_length = len(rewards)
    if episode_length == 0:
        return jnp.array([]), jnp.array([])

    advantages = jnp.zeros_like(rewards)
    next_value = final_value
    next_advantage = 0.0

    for t in reversed(range(episode_length)):
        mask = 1.0 - dones[t].astype(jnp.float32)
        td_error = rewards[t] + gamma * next_value * mask - values[t]
        advantages = advantages.at[t].set(td_error + gamma * lambda_ * next_advantage * mask)
        next_value = values[t]
        next_advantage = advantages[t]

    returns = advantages + values 
    
    adv_mean = jnp.mean(advantages)
    adv_std = jnp.std(advantages) + 1e-6 
    advantages = (advantages - adv_mean) / adv_std
    advantages = jnp.nan_to_num(advantages) 
    
    return returns, advantages


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
    
    obs_seq = traj['observations']
    actions_seq = traj['actions']
    returns_seq = traj['returns']
    advantages_seq = traj['advantages']
    old_log_probs_seq = traj['log_probs'] 
    
    
    if obs_seq.shape[0] == 0:
        return jnp.array(0.0), (jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))

    initial_hidden_state_for_scan = traj['hidden_states'][0]

    def ppo_scan_step(hidden_state_carry, scan_inputs_t):
        obs_t, action_t, old_log_prob_t, advantage_t, return_t = scan_inputs_t

        action_dist_t, next_hidden_state_carry = policy_apply(policy_params, obs_t, hidden_state_carry)
        current_log_prob_t = action_dist_t.log_prob(action_t)
        entropy_t = action_dist_t.entropy()
        
        value_t = value_apply(value_params, obs_t).squeeze()

        # PPO policy loss component (surrogate objective)
        ratio_t = jnp.exp(current_log_prob_t - old_log_prob_t) 
    
        
        adv_t_reshaped = advantage_t
        if ratio_t.ndim > advantage_t.ndim: 
            adv_t_reshaped = jnp.expand_dims(advantage_t, axis=tuple(range(advantage_t.ndim, ratio_t.ndim)))

        surr1_t = ratio_t * adv_t_reshaped
        surr2_t = jnp.clip(ratio_t, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_t_reshaped
        policy_loss_term_t = -jnp.minimum(surr1_t, surr2_t)
        
        # Value loss component
        value_loss_term_t = jnp.square(return_t - value_t)
        
        
        entropy_bonus_term_t = -entropy_t 

        
        if policy_loss_term_t.ndim > 0: policy_loss_term_t = jnp.mean(policy_loss_term_t)
        if value_loss_term_t.ndim > 0: value_loss_term_t = jnp.mean(value_loss_term_t) 
        if entropy_bonus_term_t.ndim > 0: entropy_bonus_term_t = jnp.mean(entropy_bonus_term_t)

        return next_hidden_state_carry, (policy_loss_term_t, value_loss_term_t, entropy_bonus_term_t)

    
    scan_inputs_stacked = (obs_seq, actions_seq, old_log_probs_seq, advantages_seq, returns_seq)
    
    
    _, (policy_loss_terms, value_loss_terms, entropy_bonus_terms) = jax.lax.scan(
        ppo_scan_step, initial_hidden_state_for_scan, scan_inputs_stacked
    )

    
    mean_policy_loss = jnp.mean(policy_loss_terms)
    mean_value_loss = jnp.mean(value_loss_terms)
    mean_entropy_bonus = jnp.mean(entropy_bonus_terms) 

    # Total PPO loss
    total_loss = mean_policy_loss + value_coeff * mean_value_loss + entropy_coeff * mean_entropy_bonus

    return total_loss, (mean_policy_loss, mean_value_loss, mean_entropy_bonus)


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

    def adaptation_step(i, current_params_tuple): 
        curr_policy_params, curr_value_params = current_params_tuple

        # Loss function for gradient calculation
        def loss_fn_for_grad(params_for_loss):
            p_params, v_params = params_for_loss
            
            loss_val, _ = compute_ppo_loss(
                p_params, v_params, policy_apply, value_apply, traj, clip_ratio
            )
            return loss_val

        # Calculate gradients w.r.t. (policy_params, value_params)
        adaptation_loss_val, grads_tuple = jax.value_and_grad(loss_fn_for_grad, argnums=0, has_aux=False)(
            (curr_policy_params, curr_value_params)
        )
        policy_grads, value_grads = grads_tuple

        policy_grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), policy_grads)
        value_grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), value_grads)

        new_policy_params = jax.tree_map(lambda p, g: p - inner_lr * g, curr_policy_params, policy_grads)
        new_value_params = jax.tree_map(lambda p, g: p - inner_lr * g, curr_value_params, value_grads)

        return (new_policy_params, new_value_params), adaptation_loss_val 

    initial_params_for_adaptation = (policy_params, value_params)
    
    def fori_loop_body(loop_idx, loop_state_params_tuple):
        new_params_tuple, _ = adaptation_step(loop_idx, loop_state_params_tuple)
        return new_params_tuple
   
    def adaptation_fori_body(loop_idx, current_params_tuple_fori):
        curr_policy_params_f, curr_value_params_f = current_params_tuple_fori
        def loss_fn_for_grad_f(params_for_loss_f):
            p_params_f, v_params_f = params_for_loss_f
            loss_val_f, _ = compute_ppo_loss(
                p_params_f, v_params_f, policy_apply, value_apply, traj, clip_ratio
            )
            return loss_val_f
        _, grads_tuple_f = jax.value_and_grad(loss_fn_for_grad_f)((curr_policy_params_f, curr_value_params_f))
        policy_grads_f, value_grads_f = grads_tuple_f
        policy_grads_f = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), policy_grads_f)
        value_grads_f = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), value_grads_f)
        new_policy_params_f = jax.tree_map(lambda p, g: p - inner_lr * g, curr_policy_params_f, policy_grads_f)
        new_value_params_f = jax.tree_map(lambda p, g: p - inner_lr * g, curr_value_params_f, value_grads_f)
        return (new_policy_params_f, new_value_params_f)

    final_adapted_params_tuple = lax.fori_loop(
        0, inner_steps, adaptation_fori_body, initial_params_for_adaptation
    )
    return final_adapted_params_tuple


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

    adapted_policy_params, adapted_value_params = ppo_inner_adaptation(
        init_policy_params, init_value_params,
        policy_apply, value_apply,
        train_traj, inner_lr, inner_steps, clip_ratio
    )

   
    test_loss, (policy_loss, value_loss, entropy_loss_aux) = compute_ppo_loss(
        adapted_policy_params, adapted_value_params,
        policy_apply, value_apply, test_traj, clip_ratio
    )
    return test_loss, (policy_loss, value_loss, entropy_loss_aux)


def sample_task(env, key):
    key_interf, key_fading = jax.random.split(key)
    variation = jax.random.uniform(key_interf, (), minval=0.8, maxval=1.2)
    new_max_interference = env.max_interference * variation
    
    fading_variation = jax.random.uniform(key_fading, (), minval=0.8, maxval=1.2) 
    new_fading_coherence = env.fading_coherence * fading_variation

    new_env = DynamicSpectrumEnv(
        num_bs=env.num_bs,
        num_users=env.num_users,
        num_bands=env.num_bands,
        max_steps=env.max_steps,
        max_latency=env.max_latency,
        max_power=env.max_power,
        num_power_levels=env.num_power_levels,
        power_levels=env.power_levels,
        fading_coherence=new_fading_coherence, 
        max_interference=new_max_interference,
        min_sinr=env.min_sinr
    )
    return new_env


def stack_trajectories(trajs: List[Dict]) -> Dict: 
    if not trajs: 
        return {
            'observations': jnp.array([]), 'actions': jnp.array([]), 'rewards': jnp.array([]),
            'raw_rewards': jnp.array([]), 'values': jnp.array([]), 'dones': jnp.array([], dtype=bool),
            'log_probs': jnp.array([]), 'hidden_states': [], 
            'final_value': jnp.array([]), 'final_hidden_state': [], 
            'sinr_violations': jnp.array([]), 'qos_violations': jnp.array([])
        }

    keys = trajs[0].keys()
    stacked = {}
    for k in keys:
        
        if k == 'hidden_states': 
            stacked[k] = [traj[k] for traj in trajs] 
        elif k == 'final_hidden_state': 
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
    use_wandb: bool = True,
    rollout_len: int = 50
) -> Tuple[Tuple[Any,Any], Dict[str, list]]:

    params = (policy_params, value_params)
    obs_norm_state = init_obs_normalizer(obs_dim)
    reward_norm_state = init_reward_normalizer()

    optimizer = optax.chain(
        optax.clip(1.0),
        optax.adam(meta_lr)
    )
    opt_state = optimizer.init(params)

    
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

    for iteration in range(num_iterations):
        key, iter_key = jax.random.split(key)
        task_keys_for_batch = jax.random.split(iter_key, num_tasks)

        accumulated_meta_grads = None
        total_meta_loss_this_batch = 0.0
        num_successful_tasks_in_batch = 0

        
        for task_idx in range(num_tasks):
            task_specific_key = task_keys_for_batch[task_idx]
            key_task_sample, key_train_traj, key_test_traj = jax.random.split(task_specific_key, 3) 

            task_env_instance = sample_task(env, key_task_sample)

            train_traj_raw = sample_trajectories(
                task_env_instance, params[0], params[1], policy_apply, value_apply, key_train_traj,
                obs_norm_state, reward_norm_state, num_steps=rollout_len
            )
            if train_traj_raw['observations'].shape[0] == 0: continue

            train_traj_prepared = prepare_trajectory(train_traj_raw, params[1], value_apply)

            adapted_policy_p, adapted_value_p = ppo_inner_adaptation(
                params[0], params[1], policy_apply, value_apply,
                train_traj_prepared, inner_lr, inner_steps, clip_ratio
            )

            test_traj_raw = sample_trajectories(
                task_env_instance, adapted_policy_p, adapted_value_p, policy_apply, value_apply, key_test_traj,
                obs_norm_state, reward_norm_state, num_steps=rollout_len
            )
            if test_traj_raw['observations'].shape[0] == 0: continue

            test_traj_prepared = prepare_trajectory(test_traj_raw, adapted_value_p, value_apply)

            def meta_obj_for_grad_fn(p_meta_grad):
                loss_val, _ = compute_meta_objective(
                    p_meta_grad, train_traj_prepared, test_traj_prepared,
                    policy_apply, value_apply, inner_lr, inner_steps, clip_ratio
                )
                return loss_val

            task_meta_loss_val, task_meta_grads = jax.value_and_grad(meta_obj_for_grad_fn)(params)

            total_meta_loss_this_batch += task_meta_loss_val
            if accumulated_meta_grads is None:
                accumulated_meta_grads = task_meta_grads
            else:
                accumulated_meta_grads = jax.tree_map(lambda acc_g, new_g: acc_g + new_g, accumulated_meta_grads, task_meta_grads)
            num_successful_tasks_in_batch += 1

            if train_traj_raw['observations'].shape[0] > 0:
                obs_norm_state = update_obs_normalizer(obs_norm_state, train_traj_raw['observations'])
                if 'raw_rewards' in train_traj_raw and train_traj_raw['raw_rewards'].shape[0] > 0:
                     reward_norm_state = update_reward_normalizer(reward_norm_state, train_traj_raw['raw_rewards'])
            if test_traj_raw['observations'].shape[0] > 0:
                obs_norm_state = update_obs_normalizer(obs_norm_state, test_traj_raw['observations'])
                if 'raw_rewards' in test_traj_raw and test_traj_raw['raw_rewards'].shape[0] > 0:
                    reward_norm_state = update_reward_normalizer(reward_norm_state, test_traj_raw['raw_rewards'])

        if num_successful_tasks_in_batch > 0:
            avg_meta_loss_batch = total_meta_loss_this_batch / num_successful_tasks_in_batch
            meta_losses_hist.append(float(avg_meta_loss_batch))
            mean_meta_grads = jax.tree_map(lambda g: g / num_successful_tasks_in_batch, accumulated_meta_grads)
            updates, opt_state = optimizer.update(mean_meta_grads, opt_state, params)
            params = optax.apply_updates(params, updates)
        else:
            meta_losses_hist.append(float('nan'))
            print(f"Warning: Iteration {iteration} had no successful training tasks.")

        if iteration % eval_interval == 0:
            log_payload_eval = {"iteration": iteration,
                                "meta_loss": meta_losses_hist[-1] if meta_losses_hist and not jnp.isnan(meta_losses_hist[-1]) else float('nan')}

            
            key, eval_master_key = jax.random.split(key) 
            eval_task_keys = jax.random.split(eval_master_key, num_eval_tasks)

            current_eval_pre_rewards, current_eval_post_rewards = [], []
            current_eval_pre_sinrs, current_eval_post_sinrs = [], []
            current_eval_pre_qoss, current_eval_post_qoss = [], []

            for eval_task_idx in range(num_eval_tasks):
                eval_task_specific_key = eval_task_keys[eval_task_idx]
                key_eval_task_sample, key_eval_pre_traj, key_eval_adapt, key_eval_post_traj = jax.random.split(eval_task_specific_key, 4)

                eval_task_env = sample_task(env, key_eval_task_sample)

                pre_adapt_traj_eval = sample_trajectories(
                    eval_task_env, params[0], params[1], policy_apply, value_apply, key_eval_pre_traj,
                    obs_norm_state, reward_norm_state, num_steps=rollout_len
                )
                if pre_adapt_traj_eval['observations'].shape[0] > 0:
                    current_eval_pre_rewards.append(jnp.mean(pre_adapt_traj_eval["rewards"])) 
                    current_eval_pre_sinrs.append(jnp.mean(pre_adapt_traj_eval["sinr_violations"]))
                    current_eval_pre_qoss.append(jnp.mean(pre_adapt_traj_eval["qos_violations"]))

               
                if pre_adapt_traj_eval['observations'].shape[0] > 0:
                    pre_adapt_traj_eval_prepared = prepare_trajectory(pre_adapt_traj_eval, params[1], value_apply)
                    
                    adapted_policy_p_eval, adapted_value_p_eval = ppo_inner_adaptation(
                        params[0], params[1], policy_apply, value_apply,
                        pre_adapt_traj_eval_prepared, 
                        inner_lr, inner_steps, clip_ratio 
                    )
                else: 
                    adapted_policy_p_eval, adapted_value_p_eval = params[0], params[1]


                post_adapt_traj_eval = sample_trajectories(
                    eval_task_env, adapted_policy_p_eval, adapted_value_p_eval, policy_apply, value_apply, key_eval_post_traj,
                    obs_norm_state, reward_norm_state, num_steps=rollout_len
                )
                if post_adapt_traj_eval['observations'].shape[0] > 0:
                    current_eval_post_rewards.append(jnp.mean(post_adapt_traj_eval["rewards"])) # Normalized
                    current_eval_post_sinrs.append(jnp.mean(post_adapt_traj_eval["sinr_violations"]))
                    current_eval_post_qoss.append(jnp.mean(post_adapt_traj_eval["qos_violations"]))

            # Calculate and log average metrics for evaluation interval
            if current_eval_pre_rewards: 
                apr = jnp.mean(jnp.array(current_eval_pre_rewards))
                apsr = jnp.mean(jnp.array(current_eval_post_rewards)) if current_eval_post_rewards else apr 
                asi = apsr - apr
                eval_avg_pre_rewards_hist.append(float(apr)); eval_avg_post_rewards_hist.append(float(apsr)); eval_avg_reward_improvements_hist.append(float(asi))
                log_payload_eval.update({"eval_avg_pre_reward": float(apr), "eval_avg_post_reward": float(apsr), "eval_avg_reward_improvement": float(asi)})

            if current_eval_pre_sinrs:
                aps = jnp.mean(jnp.array(current_eval_pre_sinrs))
                aops = jnp.mean(jnp.array(current_eval_post_sinrs)) if current_eval_post_sinrs else aps
                assi = aps - aops 
                eval_avg_pre_sinrs_hist.append(float(aps)); eval_avg_post_sinrs_hist.append(float(aops)); eval_avg_sinr_improvements_hist.append(float(assi))
                log_payload_eval.update({"eval_avg_pre_sinr_violation": float(aps), "eval_avg_post_sinr_violation": float(aops), "eval_avg_sinr_improvement": float(assi)})

            if current_eval_pre_qoss:
                apq = jnp.mean(jnp.array(current_eval_pre_qoss))
                apoq = jnp.mean(jnp.array(current_eval_post_qoss)) if current_eval_post_qoss else apq
                asqi = apq - apoq 
                eval_avg_pre_qoss_hist.append(float(apq)); eval_avg_post_qoss_hist.append(float(apoq)); eval_avg_qos_improvements_hist.append(float(asqi))
                log_payload_eval.update({"eval_avg_pre_qos_violation": float(apq), "eval_avg_post_qos_violation": float(apoq), "eval_avg_qos_improvement": float(asqi)})

            print(f"[Iter {iteration}] meta_loss={log_payload_eval.get('meta_loss', 0.0):.3f} "
                  f"eval_pre_r={log_payload_eval.get('eval_avg_pre_reward', 0.0):.3f} "
                  f"eval_post_r={log_payload_eval.get('eval_avg_post_reward', 0.0):.3f}")
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
    }
    return params, history_data
