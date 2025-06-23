from functools import partial
from typing import Any, Callable, Tuple, Dict
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax as tfp
from utils import flatten_state
import wandb
from utils import SpectrumState
from mLN.environment import DynamicSpectrumEnv


# Constants
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


tfd = tfp.distributions

def flatten_state(state: SpectrumState) -> jnp.ndarray:
    return jnp.concatenate([
        state.channel_gains.flatten(),
        state.interference_map.flatten(),
        state.user_latency.flatten(),
        state.spectrum_alloc.flatten(),
        state.tx_power.flatten(),
        jnp.array([state.time]),
    ])

def residual_block(x, hidden_dim):
    h = hk.Linear(hidden_dim)(x)
    h = jax.nn.relu(h)
    h = hk.Linear(hidden_dim)(h)
    return jax.nn.relu(x + h)

class MLPNetwork(hk.Module):
    def __init__(self, num_bs, num_bands, num_power_levels, hidden_dim=64, num_blocks=3):
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
        logits = jnp.clip(logits, -20.0, 20.0)
        return logits.reshape(*self.reshape_dims)

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

def init_obs_normalizer(obs_dim: int): return (jnp.zeros(obs_dim), jnp.ones(obs_dim), jnp.array(0.0))

def update_obs_normalizer(state, obs_batch):
    mean, var, count = state
    batch_mean, batch_var, batch_count = jnp.mean(obs_batch, 0), jnp.var(obs_batch, 0), obs_batch.shape[0]
    delta = batch_mean - mean
    total_count = count + batch_count
    new_mean = mean + delta * batch_count / total_count
    m_a, m_b = var * count, batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * count * batch_count / total_count
    new_var = M2 / total_count
    return (new_mean, new_var, total_count)

def normalize_obs(state, obs, clip_range=5.0):
    mean, var, count = state
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

        # Collect detailed metrics at each step
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
    obs = normalize_obs(traj['obs_norm_state'], traj['obs'])
    returns, advantages = compute_gae(traj)

    action_dist = policy_apply(policy_params, obs)
    log_probs = action_dist.log_prob(traj['actions'])
    summed_log_probs = jnp.sum(log_probs, axis=1)

    policy_loss = -jnp.mean(summed_log_probs * jax.lax.stop_gradient(advantages))
    entropy_loss = -jnp.mean(action_dist.entropy())

    value_pred = value_apply(value_params, obs)
    value_loss = jnp.mean(optax.huber_loss(value_pred, jax.lax.stop_gradient(returns)))

    combined_loss = policy_loss + config["vf_coef"] * value_loss + config["ent_coef"] * entropy_loss

    loss_info = {
        "combined_loss": combined_loss, "policy_loss": policy_loss,
        "value_loss": value_loss, "entropy": -entropy_loss
    }
    return combined_loss, loss_info

def inner_adaptation(params, policy_apply, value_apply, traj, config):
    def adaptation_step(p, _):
        policy_params, value_params = p
        (loss, loss_info), grads = jax.value_and_grad(compute_loss, argnums=(0,1), has_aux=True)(
            policy_params, value_params, policy_apply, value_apply, traj, config)

        p_grads, v_grads = grads

        new_policy_params = jax.tree.map(lambda p_i, g: p_i - config["inner_policy_lr"] * g, policy_params, p_grads)
        new_value_params = jax.tree.map(lambda v_i, g: v_i - config["inner_value_lr"] * g, value_params, v_grads)

        return (new_policy_params, new_value_params), loss_info

    (adapted_params, loss_infos) = jax.lax.scan(adaptation_step, params, None, length=config["inner_steps"])
    final_loss_info = jax.tree.map(lambda x: x[-1], loss_infos)
    return adapted_params, final_loss_info

def sample_task(env: DynamicSpectrumEnv, key: jax.random.PRNGKey):
    key, subkey1, subkey2 = jax.random.split(key, 3)
    variation = jax.random.uniform(subkey1, (), minval=0.8, maxval=1.2)
    fading_var = jax.random.uniform(subkey2, (), minval=0.8, maxval=1.2)
    return DynamicSpectrumEnv(
        fading_coherence=env.fading_coherence * fading_var,
        max_interference=env.max_interference * variation,
    )

def train_maml(config: Dict):
    env = DynamicSpectrumEnv()
    policy, value = make_networks(env.num_bs, env.num_bands, env.num_power_levels)
    key = jax.random.PRNGKey(config["seed"])
    key, p_key, v_key, t_key = jax.random.split(key, 4)
    dummy_obs = flatten_state(env.observation_spec().generate_value())
    policy_params = policy.init(p_key, dummy_obs)
    value_params = value.init(v_key, dummy_obs)
    params = (policy_params, value_params)

    optimizer = optax.chain(optax.clip_by_global_norm(config["max_grad_norm"]), optax.adam(config["meta_lr"]))
    opt_state = optimizer.init(params)

    obs_norm_state = init_obs_normalizer(dummy_obs.shape[0])

    if config["use_wandb"]:
        wandb.init(project=config["wandb_project"], name=config["wandb_name"], config=config)

    @jax.jit
    def meta_update_step(params, opt_state, obs_norm_state, meta_batch_keys):

        def compute_meta_loss(p, task_key):
            train_key, test_key = jax.random.split(task_key)
            task_env = sample_task(env, task_key)

            train_traj = sample_trajectories(task_env, p[0], p[1], policy.apply, value.apply, train_key, obs_norm_state, config["rollout_length"])

            current_obs_norm_state = update_obs_normalizer(obs_norm_state, train_traj["obs"])
            train_traj["obs_norm_state"] = current_obs_norm_state

            adapted_params, _ = inner_adaptation(p, policy.apply, value.apply, train_traj, config)

            test_traj = sample_trajectories(task_env, adapted_params[0], adapted_params[1], policy.apply, value.apply, test_key, current_obs_norm_state, config["rollout_length"])
            test_traj["obs_norm_state"] = current_obs_norm_state

            meta_loss, loss_info = compute_loss(adapted_params[0], adapted_params[1], policy.apply, value.apply, test_traj, config)

            # --- DETAILED METRIC COMPUTATION ---
            metric_keys = ["rewards", "total_throughput", "fairness_index", "sinr_violations", "latency_violations"]

            def get_avg_traj_metrics(traj):
                return {k: jnp.mean(traj[k]) for k in metric_keys}

            pre_adapt_metrics = get_avg_traj_metrics(train_traj)
            post_adapt_metrics = get_avg_traj_metrics(test_traj)

            all_metrics = {**loss_info}
            for k in metric_keys:
                pre_val = pre_adapt_metrics[k]
                post_val = post_adapt_metrics[k]
                all_metrics[f"pre_adapt_{k}"] = pre_val
                all_metrics[f"post_adapt_{k}"] = post_val
                all_metrics[f"{k}_improvement"] = post_val - pre_val

            return meta_loss, (all_metrics, current_obs_norm_state)

        value_and_grad_fn = jax.value_and_grad(compute_meta_loss, has_aux=True)
        (losses_and_aux, task_grads) = jax.vmap(value_and_grad_fn, in_axes=(None, 0))(params, meta_batch_keys)

        (_, (all_task_metrics, new_obs_norm_states)) = losses_and_aux

        avg_grads = jax.tree.map(lambda x: jnp.mean(x, axis=0), task_grads)
        # Average all collected metrics across the meta-batch
        avg_metrics = jax.tree.map(lambda x: jnp.mean(x), all_task_metrics)

        updates, new_opt_state = optimizer.update(avg_grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        final_obs_norm_state = jax.tree.map(lambda x: x[0], new_obs_norm_states)

        return new_params, new_opt_state, final_obs_norm_state, avg_metrics

    task_keys = jax.random.split(t_key, config["num_meta_iters"])
    for i in range(config["num_meta_iters"]):
        batch_keys = jax.random.split(task_keys[i], config["meta_batch_size"])
        params, opt_state, obs_norm_state, metrics = meta_update_step(
            params, opt_state, obs_norm_state, batch_keys
        )
        if i % config["log_interval"] == 0:
            # Enhanced print statement
            print(
                f"[Iter {i}] Loss: {metrics['combined_loss']:.3f}, "
                f"Pre-Reward: {metrics['pre_adapt_rewards']:.2f}, "
                f"Post-Reward: {metrics['post_adapt_rewards']:.2f}, "
                f"Improvement: {metrics['rewards_improvement']:.2f}, "
                f"Post-Throughput: {metrics['post_adapt_total_throughput']:.2f}, "
                f"Post-SINR-Violations: {metrics['post_adapt_sinr_violations']:.2f}"
            )
            # Log all metrics to wandb
            if config["use_wandb"]:
                wandb.log({"iteration": i, **metrics})

    if config["use_wandb"]:
        wandb.finish()

    return params, metrics
   
