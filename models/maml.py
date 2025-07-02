from functools import partial
from typing import Any, Callable, Tuple, Dict, List
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax as tfp
from collections import namedtuple
import wandb
from utils import flatten_state, SpectrumState
from mLN.environment import DynamicSpectrumEnv

NUM_BS = 3  
NUM_BANDS = 4  
NUM_USERS = 5  
NUM_POWER_LEVELS = 5  
META_LR = 1e-3
INNER_LR = 0.1
META_BATCH_SIZE = 4
NUM_INNER_STEPS = 1
NUM_META_ITERS = 1000
ROLLOUT_LENGTH = 50
DISCOUNT_FACTOR = 0.99  
NUM_META_BATCHES = 10

# ==============================================================================
# CONSTANTS
# ==============================================================================
ROLLOUT_LENGTH = 50 
BANDWIDTH_HZ = 10e6
NOISE_FIGURE_DB = 7.0


tfd = tfp.distributions

def residual_block(x, hidden_dim):
    h = hk.Linear(hidden_dim)(x)
    h = jax.nn.relu(h)
    h = hk.Linear(hidden_dim)(h)
    return jax.nn.relu(x + h)

class MLPNetwork(hk.Module):
    def __init__(self, num_bs, num_bands, num_power_levels, hidden_dim=128, num_blocks=3):
        super().__init__()
        self.output_size = num_bs * num_bands * num_power_levels
        self.reshape_dims = (-1, num_bs * num_bands, num_power_levels)
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

    def __call__(self, x):
        x = hk.Flatten()(x)
        x = hk.Linear(self.hidden_dim)(x)
        x = jax.nn.relu(x)
        for _ in range(self.num_blocks):
            x = residual_block(x, self.hidden_dim)
        logits = hk.Linear(self.output_size)(x)
        logits = jnp.clip(logits, -10.0, 10.0) 
        return logits.reshape(*self.reshape_dims)

class ValueNetwork(hk.Module):
    def __init__(self, hidden_dim=128, num_blocks=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

    def __call__(self, x):
        x = hk.Flatten()(x)
        x = hk.Linear(self.hidden_dim)(x)
        x = jax.nn.relu(x)
        for _ in range(self.num_blocks):
            x = residual_block(x, self.hidden_dim)
        return hk.Linear(1)(x).squeeze(axis=-1)

def make_networks(num_bs, num_bands, num_power_levels):
    def policy_fn_builder(obs):
        net = MLPNetwork(num_bs, num_bands, num_power_levels)
        logits = net(obs)
        return tfd.Categorical(logits=logits)

    def value_fn_builder(obs):
        net = ValueNetwork()
        return net(obs)

    policy = hk.without_apply_rng(hk.transform(policy_fn_builder))
    value = hk.without_apply_rng(hk.transform(value_fn_builder))
    return policy, value

ObsNormalizerState = tuple

def init_obs_normalizer(obs_dim: int) -> ObsNormalizerState:
    return (jnp.zeros(obs_dim), jnp.ones(obs_dim), jnp.array(0.0))

def update_obs_normalizer(state: ObsNormalizerState, obs_batch: jnp.ndarray) -> ObsNormalizerState:
    """Updates the observation normalizer's statistics using Welford's algorithm."""
    mean, var, count = state
    batch_mean, batch_var, batch_count = jnp.mean(obs_batch, 0), jnp.var(obs_batch, 0), float(obs_batch.shape[0])
    delta = batch_mean - mean
    total_count = count + batch_count


    def update_fn(op):
        mean_op, var_op, count_op, delta_op, batch_var_op, batch_count_op, total_count_op = op
        
        new_mean = mean_op + delta_op * batch_count_op / total_count_op
        
        m_a = var_op * count_op
        m_b = batch_var_op * batch_count_op
        M2 = m_a + m_b + jnp.square(delta_op) * count_op * batch_count_op / total_count_op
        new_var = M2 / total_count_op
        return new_mean, new_var, total_count_op

    def identity_fn(op):
        mean_op, _, _, _, batch_var_op, _, total_count_op = op
        return batch_mean, batch_var, total_count_op

    
    operands = (mean, var, count, delta, batch_var, batch_count, total_count)
    
   
    new_mean, new_var, new_count = jax.lax.cond(
        total_count > 1e-8, 
        update_fn,
        identity_fn,
        operands
    )
    
    return (new_mean, new_var, new_count)


def normalize_obs(state: ObsNormalizerState, obs: jnp.ndarray, clip_range=5.0) -> jnp.ndarray:
    mean, var, _ = state
    normed = (obs - mean) / jnp.sqrt(var + 1e-8)
    return jnp.clip(normed, -clip_range, clip_range)

def sample_trajectories(
    task_env: DynamicSpectrumEnv, policy_params: Any, value_params: Any,
    policy_apply: Callable, value_apply: Callable, key: jax.random.PRNGKey,
    obs_norm_state: ObsNormalizerState, num_steps: int
) -> Dict[str, jnp.ndarray]:

    state, timestep = task_env.reset(key)

    def scan_fn(carry, _):
        state, key = carry
        obs = flatten_state(state)
        norm_obs = normalize_obs(obs_norm_state, obs)
        key, action_key = jax.random.split(key)
        dist = policy_apply(policy_params, norm_obs)
        action = dist.sample(seed=action_key).reshape(-1)
        value = value_apply(value_params, norm_obs)
        next_state, next_timestep = task_env.step(state, action)

        step_metrics = task_env._compute_metrics(next_state)

        out = {
            "obs": obs, "actions": action, "rewards": next_timestep.reward,
            "values": value, "dones": next_timestep.last(),
            **step_metrics
        }
        return (next_state, key), out

    (final_state, _), traj = jax.lax.scan(scan_fn, (state, key), None, length=num_steps)
    final_obs = normalize_obs(obs_norm_state, flatten_state(final_state))
    traj["final_value"] = value_apply(value_params, final_obs)
    return traj

def compute_gae(traj, gamma=0.99, lambda_=0.95):
    def gae_step(carry, transition):
        gae, next_val = carry
        reward, done, value = transition
        delta = reward + gamma * next_val * (1.0 - done) - value
        gae = delta + gamma * lambda_ * (1.0 - done) * gae
        return (gae, jax.lax.stop_gradient(value)), gae

    _, advantages = jax.lax.scan(gae_step, (0.0, traj["final_value"]),
                                 (traj["rewards"], traj["dones"], traj["values"]), reverse=True)

    adv_mean = jnp.mean(advantages)
    adv_std = jnp.std(advantages) + 1e-8
    advantages = (advantages - adv_mean) / adv_std

    returns = advantages + traj["values"]
    return returns, advantages

def compute_loss(policy_params, value_params, policy_apply, value_apply, traj, config):
    vf_coef = config.vf_coef if hasattr(config, 'vf_coef') else config["vf_coef"]
    ent_coef = config.ent_coef if hasattr(config, 'ent_coef') else config["ent_coef"]
    
    obs = normalize_obs(traj['obs_norm_state'], traj['obs'])
    returns, advantages = compute_gae(traj)

    action_dist = policy_apply(policy_params, obs)
    log_probs = action_dist.log_prob(traj['actions'])
    summed_log_probs = jnp.sum(log_probs, axis=1)

    policy_loss = -jnp.mean(summed_log_probs * jax.lax.stop_gradient(advantages))
    
    entropy_loss = -jnp.mean(jnp.sum(action_dist.entropy(), axis=-1))

    value_pred = value_apply(value_params, obs)
    value_loss = jnp.mean(optax.huber_loss(value_pred, jax.lax.stop_gradient(returns)))

    combined_loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

    loss_info = {
        "combined_loss": combined_loss, "policy_loss": policy_loss,
        "value_loss": value_loss, "entropy": -entropy_loss, "mean_reward": jnp.mean(traj["rewards"]),
        "mean_sinr_violations": jnp.mean(traj["sinr_violations"]),
        "mean_latency_violations": jnp.mean(traj["latency_violations"])
    }
    return combined_loss, loss_info

def inner_adaptation(params, policy_apply, value_apply, traj, config):
    inner_policy_lr = config.inner_policy_lr if hasattr(config, 'inner_policy_lr') else config["inner_policy_lr"]
    inner_value_lr = config.inner_value_lr if hasattr(config, 'inner_value_lr') else config["inner_value_lr"]
    inner_steps = config.inner_steps if hasattr(config, 'inner_steps') else config["inner_steps"]

    def adaptation_step(p, _):
        policy_params, value_params = p
        (loss, loss_info), grads = jax.value_and_grad(compute_loss, argnums=(0,1), has_aux=True)(
            policy_params, value_params, policy_apply, value_apply, traj, config)

        p_grads, v_grads = grads

        new_policy_params = jax.tree.map(lambda p_i, g: p_i - inner_policy_lr * g, policy_params, p_grads)
        new_value_params = jax.tree.map(lambda v_i, g: v_i - inner_value_lr * g, value_params, v_grads)

        return (new_policy_params, new_value_params), loss_info

    (adapted_params, loss_infos) = jax.lax.scan(adaptation_step, params, None, length=inner_steps)
    final_loss_info = jax.tree.map(lambda x: x[-1], loss_infos)
    return adapted_params, final_loss_info

def sample_task(env: DynamicSpectrumEnv, key: jax.random.PRNGKey):
    key, subkey1, subkey2 = jax.random.split(key, 3)
    variation = jax.random.uniform(subkey1, (), minval=0.8, maxval=1.2)
    fading_var = jax.random.uniform(subkey2, (), minval=0.8, maxval=1.2)
    return DynamicSpectrumEnv(
        fading_coherence=env.fading_coherence * fading_var,
        max_interference=env.max_interference * variation,
        num_bs=env.num_bs, num_users=env.num_users, num_bands=env.num_bands,
        max_steps=env.max_steps, max_latency=env.max_latency,
        num_power_levels=env.num_power_levels, power_levels=env.power_levels,
        min_sinr=env.min_sinr
    )

@partial(jax.jit, static_argnames=("env", "policy_apply", "value_apply", "config"))
def evaluate_adaptation(
    params: Tuple[Any, Any],
    obs_norm_state: ObsNormalizerState,
    key: jax.random.PRNGKey,
    env: DynamicSpectrumEnv,
    policy_apply: Callable,
    value_apply: Callable,
    config: Any,
):
    """Runs evaluation on a batch of tasks to measure pre- and post-adaptation performance."""
    eval_task_keys = jax.random.split(key, config.num_eval_tasks)

    def run_single_task_eval(task_key):
        task_env = sample_task(env, task_key)
        p_policy, p_value = params
        pre_key, adapt_key, post_key = jax.random.split(task_key, 3)

        pre_traj = sample_trajectories(task_env, p_policy, p_value, policy_apply, value_apply, pre_key, obs_norm_state, config.rollout_length)

        adapt_traj = sample_trajectories(task_env, p_policy, p_value, policy_apply, value_apply, adapt_key, obs_norm_state, config.rollout_length)
        adapt_traj["obs_norm_state"] = obs_norm_state
        adapted_params, _ = inner_adaptation(params, policy_apply, value_apply, adapt_traj, config)
        adapted_p_policy, adapted_p_value = adapted_params

        post_traj = sample_trajectories(task_env, adapted_p_policy, adapted_p_value, policy_apply, value_apply, post_key, obs_norm_state, config.rollout_length)

        metric_keys = ["rewards", "total_throughput", "fairness_index", "sinr_violations", "latency_violations"]
        
        def get_avg_traj_metrics(traj, prefix):
            return {f"{prefix}_{k}": jnp.mean(traj[k]) for k in metric_keys}

        pre_metrics = get_avg_traj_metrics(pre_traj, "pre")
        post_metrics = get_avg_traj_metrics(post_traj, "post")
        return {**pre_metrics, **post_metrics}

    all_tasks_metrics = jax.vmap(run_single_task_eval)(eval_task_keys)
    avg_metrics = jax.tree.map(lambda x: jnp.mean(x, axis=0), all_tasks_metrics)

    final_eval_metrics = {}
    metric_keys_gain = ["rewards", "total_throughput", "fairness_index"]
    metric_keys_reduce = ["sinr_violations", "latency_violations"]

    for k in metric_keys_gain:
        pre_k, post_k = f"pre_{k}", f"post_{k}"
        final_eval_metrics[f"eval_{pre_k}"] = avg_metrics[pre_k]
        final_eval_metrics[f"eval_{post_k}"] = avg_metrics[post_k]
        final_eval_metrics[f"eval_{k}_improvement"] = avg_metrics[post_k] - avg_metrics[pre_k]
    
    for k in metric_keys_reduce:
        pre_k, post_k = f"pre_{k}", f"post_{k}"
        final_eval_metrics[f"eval_{pre_k}"] = avg_metrics[pre_k]
        final_eval_metrics[f"eval_{post_k}"] = avg_metrics[post_k]
        final_eval_metrics[f"eval_{k}_improvement"] = avg_metrics[pre_k] - avg_metrics[post_k]

    return final_eval_metrics

def train_maml(config: Dict) -> Tuple[Any, List[Dict]]:
    Config = namedtuple("Config", config.keys())
    h_config = Config(**config)

    env = DynamicSpectrumEnv()
    policy, value = make_networks(env.num_bs, env.num_bands, env.num_power_levels)
    key = jax.random.PRNGKey(h_config.seed)
    key, p_key, v_key = jax.random.split(key, 3)
    dummy_obs = flatten_state(env.observation_spec().generate_value())
    policy_params = policy.init(p_key, dummy_obs)
    value_params = value.init(v_key, dummy_obs)
    params = (policy_params, value_params)

    optimizer = optax.chain(optax.clip_by_global_norm(h_config.max_grad_norm), optax.adam(h_config.meta_lr))
    opt_state = optimizer.init(params)

    obs_norm_state = init_obs_normalizer(dummy_obs.shape[0])

    if h_config.use_wandb:
        wandb.init(project=h_config.wandb_project, name=h_config.wandb_name, config=config)
    
    jitted_eval_fn = evaluate_adaptation

    @jax.jit
    def meta_update_step(p_step, opt_s_step, obs_norm_s_step, meta_batch_keys):

        def compute_meta_loss(p_inner, task_key):
            train_key, test_key = jax.random.split(task_key)
            task_env = sample_task(env, task_key)
            train_traj = sample_trajectories(task_env, p_inner[0], p_inner[1], policy.apply, value.apply, train_key, obs_norm_s_step, h_config.rollout_length)
            current_obs_norm_state = update_obs_normalizer(obs_norm_s_step, train_traj["obs"])
            train_traj["obs_norm_state"] = current_obs_norm_state

            adapted_params, _ = inner_adaptation(p_inner, policy.apply, value.apply, train_traj, h_config)

            test_traj = sample_trajectories(task_env, adapted_params[0], adapted_params[1], policy.apply, value.apply, test_key, current_obs_norm_state, h_config.rollout_length)
            test_traj["obs_norm_state"] = current_obs_norm_state
            
            meta_loss, loss_info = compute_loss(adapted_params[0], adapted_params[1], policy.apply, value.apply, test_traj, h_config)
            
            metric_keys = ["rewards", "total_throughput", "fairness_index", "sinr_violations", "latency_violations"]
            def get_avg_traj_metrics(traj):
                return {k: jnp.mean(traj[k]) for k in metric_keys}

            pre_adapt_metrics = get_avg_traj_metrics(train_traj)
            post_adapt_metrics = get_avg_traj_metrics(test_traj)

            all_metrics = {**loss_info}
            for k in metric_keys:
                pre_val, post_val = pre_adapt_metrics[k], post_adapt_metrics[k]
                all_metrics[f"train_pre_adapt_{k}"] = pre_val
                all_metrics[f"train_post_adapt_{k}"] = post_val
                improvement = post_val - pre_val if k not in ["sinr_violations", "latency_violations"] else pre_val - post_val
                all_metrics[f"train_{k}_improvement"] = improvement
            return meta_loss, (all_metrics, train_traj["obs"])

        value_and_grad_fn = jax.value_and_grad(compute_meta_loss, has_aux=True)
        (losses_and_aux, task_grads) = jax.vmap(value_and_grad_fn, in_axes=(None, 0))(p_step, meta_batch_keys)

        (_, (all_task_metrics, all_task_train_obs)) = losses_and_aux
        
        avg_grads = jax.tree.map(lambda x: jnp.mean(x, axis=0), task_grads)
        avg_metrics = jax.tree.map(lambda x: jnp.mean(x, axis=0), all_task_metrics)
        
        updates, new_opt_state = optimizer.update(avg_grads, opt_s_step, p_step)
        new_params = optax.apply_updates(p_step, updates)
        obs_for_global_update = jnp.reshape(all_task_train_obs, (-1, all_task_train_obs.shape[-1]))
        final_obs_norm_state = update_obs_normalizer(obs_norm_s_step, obs_for_global_update)

        return new_params, new_opt_state, final_obs_norm_state, avg_metrics

    # Main training loop
    history = []
    for i in range(h_config.num_meta_iters):
        decay_progress = jnp.clip(i / h_config.ent_coef_decay_steps, 0.0, 1.0)
        current_ent_coef = h_config.ent_coef_start * (1 - decay_progress) + h_config.ent_coef_end * decay_progress
        
        iter_config_dict = h_config._asdict()
        iter_config_dict['ent_coef'] = current_ent_coef 
        IterConfig = namedtuple("IterConfig", iter_config_dict.keys())
        iter_h_config = IterConfig(**iter_config_dict)
        key, train_key, eval_key = jax.random.split(key, 3)
        
        batch_keys = jax.random.split(train_key, h_config.meta_batch_size)
        params, opt_state, obs_norm_state, train_metrics = meta_update_step(
            params, opt_state, obs_norm_state, batch_keys
        )
        history.append(train_metrics)

        if i % h_config.log_interval == 0:
            print(
                f"[Iter {i:04d}] Loss: {train_metrics['combined_loss']:.3f}, P-Loss: {train_metrics['policy_loss']:.3f}, "
                f"V-Loss: {train_metrics['value_loss']:.3f}, Entropy: {train_metrics['entropy']:.3f}, "
                f"Train Reward (Post-Adapt): {train_metrics['train_post_adapt_rewards']:.3f}, "
                f"Train Reward: {train_metrics['mean_reward']:.3f}"
            )
            if h_config.use_wandb:
                wandb.log({"iteration": i, **train_metrics})
        
        if i > 0 and i % h_config.eval_interval == 0:
            eval_metrics = jitted_eval_fn(
                params, obs_norm_state, eval_key, env,
                policy.apply, value.apply, h_config
            )
            print(
                f"[Iter {i:04d} EVAL] Pre-Reward: {eval_metrics['eval_pre_rewards']:.3f}, "
                f"Post-Reward: {eval_metrics['eval_post_rewards']:.3f} "
                f"(Impr: {eval_metrics['eval_rewards_improvement']:.3f})"
            )
            if h_config.use_wandb:
                wandb.log({"iteration": i, **eval_metrics})

    if h_config.use_wandb:
        wandb.finish()

    return params, history
