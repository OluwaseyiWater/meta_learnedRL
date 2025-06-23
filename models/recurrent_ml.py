import jax
import jax.numpy as jnp
import optax
import haiku as hk
import chex
import wandb
from functools import partial
from typing import Any, Callable, Tuple, Dict, Optional, Sequence
from collections import namedtuple
from mLN.environment import DynamicSpectrumEnv
from utils import flatten_state, SpectrumState
import tensorflow_probability.substrates.jax as tfp


tfd = tfp.distributions

def compute_gae(traj: Dict, gamma: float, lambda_: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    last_value = traj["final_value"]
    reversed_traj_parts = jax.tree.map(lambda x: x[::-1], (traj["rewards"], traj["dones"], traj["values"]))

    def gae_step(carry, inputs):
        gae_plus_1, value_plus_1 = carry
        reward_t, done_t, value_t = inputs
        delta_t = reward_t + gamma * (1.0 - done_t) * value_plus_1 - value_t
        gae_t = delta_t + gamma * lambda_ * (1.0 - done_t) * gae_plus_1
        return (gae_t, value_t), gae_t

    _, advantages_reversed = jax.lax.scan(gae_step, (0.0, last_value), reversed_traj_parts)
    advantages = advantages_reversed[::-1]
    returns = advantages + traj["values"]
    return jax.lax.stop_gradient(returns), jax.lax.stop_gradient(advantages)

def init_obs_normalizer(dim): return (jnp.zeros(dim), jnp.ones(dim), 1e-8)
def update_obs_normalizer(state, obs_batch): return state
def normalize_obs(state, obs): return (obs - state[0]) / jnp.sqrt(state[1] + state[2])

# --- 1. Recurrent Network Definitions ---

class RecurrentActor(hk.Module):
    def __init__(self, lstm_hidden_dim: int, mlp_hidden_dim: int, num_outputs: int):
        super().__init__()
        self.lstm = hk.LSTM(lstm_hidden_dim)
        self.body = hk.nets.MLP([mlp_hidden_dim, mlp_hidden_dim])
        self.head = hk.Linear(num_outputs)
    def __call__(self, obs: chex.Array, state: Optional[hk.LSTMState]) -> Tuple[chex.Array, hk.LSTMState]:
        batch_size = obs.shape[0]
        if state is None: state = self.lstm.initial_state(batch_size)
        embedding, new_state = self.lstm(obs, state)
        return jnp.clip(self.head(self.body(embedding)), -10.0, 10.0), new_state

class Critic(hk.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.body = hk.nets.MLP([hidden_dim, hidden_dim])
        self.head = hk.Linear(1)
    def __call__(self, obs: chex.Array) -> chex.Array:
        return self.head(self.body(hk.Flatten()(obs))).squeeze(axis=-1)

def make_recurrent_networks(num_outputs: int, config: Any):
    """MODIFIED to take config for network dimensions."""
    def actor_fn(o, s):
        net = RecurrentActor(
            lstm_hidden_dim=config.lstm_hidden_dim,
            mlp_hidden_dim=config.mlp_hidden_dim,
            num_outputs=num_outputs
        )
        return net(o, s)

    def critic_fn(o):
        net = Critic(hidden_dim=config.critic_hidden_dim)
        return net(o)

    actor = hk.without_apply_rng(hk.transform(actor_fn))
    critic = hk.without_apply_rng(hk.transform(critic_fn))
    return actor, critic

# --- 2. Trajectory Sampling and PPO Loss (MODIFIED FOR METRICS) ---

def _get_env_metrics(state: SpectrumState, env: DynamicSpectrumEnv) -> Dict[str, chex.Array]:
    """Helper to compute environment-specific metrics from a state."""
    sinr_db = env._compute_sinr(state)
    sinr_violations = jnp.sum(sinr_db < env.min_sinr)
    latency_violations = jnp.sum(state.user_latency > env.max_latency)
    return {"sinr_violations": sinr_violations, "latency_violations": latency_violations}

def sample_recurrent_trajectories(
    env: DynamicSpectrumEnv, actor_params: Any, critic_params: Any,
    actor_apply: Callable, critic_apply: Callable, key: jax.random.PRNGKey,
    num_steps: int, norm_state: Tuple, norm_fn: Callable, num_actions: int, num_action_values: int
) -> Dict[str, jnp.ndarray]:
    """MODIFIED to also collect environment-specific metrics."""
    state, _ = env.reset(key)
    initial_obs = flatten_state(state)[None, :]
    norm_initial_obs = norm_fn(norm_state, initial_obs)
    _, initial_h_state = actor_apply(actor_params, norm_initial_obs, None)

    def scan_fn(carry, _):
        env_state, h_state, key = carry
        obs = flatten_state(env_state)
        norm_obs = norm_fn(norm_state, obs)
        key, action_key = jax.random.split(key)
        logits, new_h_state = actor_apply(actor_params, norm_obs[None, :], h_state)
        dist = tfd.Categorical(logits=logits.reshape(1, num_actions, num_action_values))
        action = dist.sample(seed=action_key)
        log_prob = dist.log_prob(action)
        value = critic_apply(critic_params, norm_obs)
        next_state, next_timestep = env.step(env_state, action.squeeze(0))
        env_metrics = _get_env_metrics(next_state, env) # <-- Collect metrics
        out = {"obs": obs, "actions": action.squeeze(0), "rewards": next_timestep.reward,
               "values": value, "dones": next_timestep.last(), "log_probs": log_prob.squeeze(0),
               **env_metrics} # <-- Add to output
        return (next_state, new_h_state, key), out

    (final_env_state, _, _), traj = jax.lax.scan(scan_fn, (state, initial_h_state, key), None, length=num_steps)
    new_norm_state = update_obs_normalizer(norm_state, traj["obs"][None, :])
    final_obs_flat = flatten_state(final_env_state)
    final_norm_obs = norm_fn(new_norm_state, final_obs_flat)
    traj["final_value"] = critic_apply(critic_params, final_norm_obs)
    traj["obs_norm_state"] = new_norm_state
    return traj

def compute_ppo_loss(
    actor_params: Any, critic_params: Any, actor_apply: Callable, critic_apply: Callable,
    traj: Dict, old_log_probs: chex.Array, norm_state: Tuple, norm_fn: Callable,
    config: Any, num_actions: int, num_action_values: int
) -> Tuple[chex.Array, Dict]:
    """MODIFIED to return additional metrics."""
    obs = norm_fn(norm_state, traj['obs'])
    returns, advantages = compute_gae(traj, config.gamma, config.lambda_)
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    
    def scan_fn(h_state, inputs):
        obs_t, action_t, done_t = inputs
        h_state = jax.tree.map(lambda x: jnp.where(done_t, jnp.zeros_like(x), x), h_state)
        logits, new_h_state = actor_apply(actor_params, obs_t[None, :], h_state)
        dist = tfd.Categorical(logits=logits.reshape(1, num_actions, num_action_values))
        log_prob = dist.log_prob(action_t[None, :])
        entropy = dist.entropy()
        return new_h_state, (log_prob.squeeze(0), entropy.squeeze(0))
    
    _, initial_h_state = actor_apply(actor_params, obs[0][None, :], None)
    dones_for_reset = jnp.insert(traj['dones'][:-1], 0, True)
    _, (log_probs, entropies) = jax.lax.scan(scan_fn, initial_h_state, (obs, traj['actions'], dones_for_reset))
    
    values = jax.vmap(critic_apply, in_axes=(None, 0))(critic_params, obs)
    summed_log_probs = jnp.sum(log_probs, axis=1)
    summed_old_log_probs = jax.lax.stop_gradient(jnp.sum(old_log_probs, axis=1))
    ratio = jnp.exp(summed_log_probs - summed_old_log_probs)
    clipped_ratio = jnp.clip(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
    
    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))
    entropy_loss = -jnp.mean(jnp.sum(entropies, axis=1))
    value_loss = jnp.mean(optax.huber_loss(values, jax.lax.stop_gradient(returns)))
    combined_loss = policy_loss + config.vf_coef * value_loss + config.ent_coef * entropy_loss
    
    # MODIFIED loss_info dict
    loss_info = {
        "combined_loss": combined_loss, "policy_loss": policy_loss, "value_loss": value_loss,
        "entropy": -entropy_loss, "mean_reward": jnp.mean(traj["rewards"]),
        "mean_sinr_violations": jnp.mean(traj["sinr_violations"]),
        "mean_latency_violations": jnp.mean(traj["latency_violations"])
    }
    return combined_loss, loss_info

# --- 3. MAML-PPO Training Loop (REFACTORED) ---

@partial(jax.jit, static_argnames=("env", "actor", "critic", "config", "num_actions", "num_action_values", "norm_fn"))
def evaluate_adaptation(
    params: Tuple[Any, Any],
    norm_state: Tuple,
    key: jax.random.PRNGKey,
    env: DynamicSpectrumEnv,
    actor: Any,
    critic: Any,
    config: Any,
    num_actions: int,
    num_action_values: int,
    norm_fn: Callable,
):
    """JIT-compiled and vmapped evaluation for a single task."""
    actor_params, critic_params = params
    pre_key, adapt_key, post_key = jax.random.split(key, 3)

    pre_traj = sample_recurrent_trajectories(
        env, actor_params, critic_params, actor.apply, critic.apply, pre_key,
        config.rollout_length, norm_state, norm_fn, num_actions, num_action_values)
    pre_metrics = {
        "reward": jnp.mean(pre_traj["rewards"]),
        "sinr_violations": jnp.mean(pre_traj["sinr_violations"]),
        "latency_violations": jnp.mean(pre_traj["latency_violations"]),
    }

    def ppo_inner_update(inner_p, inner_key):
        inner_actor_p, inner_critic_p = inner_p
        train_traj = sample_recurrent_trajectories(
            env, inner_actor_p, inner_critic_p, actor.apply, critic.apply, inner_key,
            config.rollout_length, norm_state, norm_fn, num_actions, num_action_values)
        ppo_grad_fn = jax.grad(compute_ppo_loss, argnums=(0, 1), has_aux=True)
        (actor_grad, critic_grad), _ = ppo_grad_fn(
            inner_actor_p, inner_critic_p, actor.apply, critic.apply, train_traj,
            train_traj['log_probs'], norm_state, norm_fn, config, num_actions, num_action_values)
        new_actor_p = jax.tree.map(lambda p, g: p - config.inner_lr * g, inner_actor_p, actor_grad)
        new_critic_p = jax.tree.map(lambda p, g: p - config.inner_lr * g, inner_critic_p, critic_grad)
        return (new_actor_p, new_critic_p), None

    adapt_keys = jax.random.split(adapt_key, config.inner_steps)
    adapted_params, _ = jax.lax.scan(ppo_inner_update, params, adapt_keys)
    adapted_actor_p, adapted_critic_p = adapted_params

    post_traj = sample_recurrent_trajectories(
        env, adapted_actor_p, adapted_critic_p, actor.apply, critic.apply, post_key,
        config.rollout_length, norm_state, norm_fn, num_actions, num_action_values)
    post_metrics = {
        "reward": jnp.mean(post_traj["rewards"]),
        "sinr_violations": jnp.mean(post_traj["sinr_violations"]),
        "latency_violations": jnp.mean(post_traj["latency_violations"]),
    }
    return pre_metrics, post_metrics

def train_recurrent_maml_ppo(config: Dict) -> Tuple[Any, Dict]:
    Config = namedtuple("Config", config.keys())
    h_config = Config(**config)

    env = DynamicSpectrumEnv()
    num_actions = env.num_bs * env.num_bands
    num_action_values = env.num_power_levels
    num_outputs = num_actions * num_action_values
    actor, critic = make_recurrent_networks(num_outputs, h_config)
    
    key = jax.random.PRNGKey(h_config.seed)
    key, p_key, v_key, t_key = jax.random.split(key, 4)
    
    dummy_state = env.observation_spec().generate_value()
    dummy_obs = flatten_state(dummy_state)[None, :]
    
    actor_params = actor.init(p_key, dummy_obs, None)
    critic_params = critic.init(v_key, dummy_obs.squeeze(0))
    params = (actor_params, critic_params)
    
    optimizer = optax.chain(optax.clip_by_global_norm(h_config.max_grad_norm), optax.adam(h_config.meta_lr))
    opt_state = optimizer.init(params)
    norm_state = init_obs_normalizer(dummy_obs.shape[-1])

    if h_config.use_wandb:
        try:
            wandb.init(project=h_config.wandb_project, name=h_config.wandb_name, config=config)
        except Exception as e: print(f"Warning: Failed to initialize WandB: {e}")

    @partial(jax.jit, static_argnames=("actor", "critic", "optimizer", "config", "num_actions", "num_action_values", "norm_fn"))
    def meta_update_step(params, opt_state, norm_state, task_key, actor, critic, optimizer, config, num_actions, num_action_values, norm_fn):
        def compute_meta_loss(p, batch_key):
            def ppo_inner_update(inner_p, inner_key):
                inner_actor_p, inner_critic_p = inner_p
                train_traj = sample_recurrent_trajectories(
                    env, inner_actor_p, inner_critic_p, actor.apply, critic.apply, inner_key,
                    config.rollout_length, norm_state, norm_fn, num_actions, num_action_values)
                ppo_grad_fn = jax.grad(compute_ppo_loss, argnums=(0, 1), has_aux=True)
                (actor_grad, critic_grad), _ = ppo_grad_fn(
                    inner_actor_p, inner_critic_p, actor.apply, critic.apply, train_traj,
                    train_traj['log_probs'], norm_state, norm_fn, config, num_actions, num_action_values)
                new_actor_p = jax.tree.map(lambda p, g: p - config.inner_lr * g, inner_actor_p, actor_grad)
                new_critic_p = jax.tree.map(lambda p, g: p - config.inner_lr * g, inner_critic_p, critic_grad)
                return (new_actor_p, new_critic_p), None

            adapt_keys = jax.random.split(batch_key, config.inner_steps)
            adapted_params, _ = jax.lax.scan(ppo_inner_update, p, adapt_keys)
            adapted_actor_p, adapted_critic_p = adapted_params
            test_traj = sample_recurrent_trajectories(
                env, adapted_actor_p, adapted_critic_p, actor.apply, critic.apply, batch_key,
                config.rollout_length, norm_state, norm_fn, num_actions, num_action_values)
            meta_loss, loss_info = compute_ppo_loss(
                adapted_actor_p, adapted_critic_p, actor.apply, critic.apply, test_traj,
                test_traj['log_probs'], norm_state, norm_fn, config, num_actions, num_action_values)
            return meta_loss, (loss_info, test_traj["obs_norm_state"])

        value_and_grad_fn = jax.value_and_grad(compute_meta_loss, has_aux=True)
        task_keys = jax.random.split(task_key, config.meta_batch_size)
        (losses_and_aux, task_grads) = jax.vmap(value_and_grad_fn, in_axes=(None, 0))(params, task_keys)
        
        (_, (loss_infos, new_norm_states)) = losses_and_aux
        avg_grads = jax.tree.map(lambda x: jnp.mean(x, axis=0), task_grads)
        avg_metrics = jax.tree.map(lambda x: jnp.mean(x, axis=0), loss_infos)
        final_norm_state = jax.tree.map(lambda x: jnp.mean(x, axis=0), new_norm_states)
        
        updates, new_opt_state = optimizer.update(avg_grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, final_norm_state, avg_metrics

    all_metrics = []
    main_keys = jax.random.split(t_key, h_config.num_meta_iters)
    
    for i in range(h_config.num_meta_iters):
        params, opt_state, norm_state, train_metrics = meta_update_step(
            params, opt_state, norm_state, main_keys[i],
            actor, critic, optimizer, h_config, num_actions, num_action_values, normalize_obs
        )
        
        if i % h_config.log_interval == 0:
            print(f"[Iter {i:04d}] Loss: {train_metrics['combined_loss']:.3f}, P-Loss: {train_metrics['policy_loss']:.3f}, "
                  f"V-Loss: {train_metrics['value_loss']:.3f}, Entropy: {train_metrics['entropy']:.3f}, "
                  f"Train Reward: {train_metrics['mean_reward']:.3f}")
            
            if h_config.use_wandb:
                wandb.log({"iteration": i, **train_metrics})

        if i % h_config.eval_interval == 0:
            eval_keys = jax.random.split(jax.random.PRNGKey(i), h_config.num_eval_tasks)
            vmapped_eval = jax.vmap(evaluate_adaptation, in_axes=(None, None, 0, None, None, None, None, None, None, None))
            pre_metrics_batch, post_metrics_batch = vmapped_eval(
                params, norm_state, eval_keys, env, actor, critic, h_config,
                num_actions, num_action_values, normalize_obs
            )
            avg_pre_metrics = jax.tree.map(lambda x: x.mean(), pre_metrics_batch)
            avg_post_metrics = jax.tree.map(lambda x: x.mean(), post_metrics_batch)
            
            eval_summary = {
                "avg_pre_reward": avg_pre_metrics["reward"], "avg_pre_sinr_violations": avg_pre_metrics["sinr_violations"], "avg_pre_latency_violations": avg_pre_metrics["latency_violations"],
                "avg_post_reward": avg_post_metrics["reward"], "avg_post_sinr_violations": avg_post_metrics["sinr_violations"], "avg_post_latency_violations": avg_post_metrics["latency_violations"],
                "reward_improvement": avg_post_metrics["reward"] - avg_pre_metrics["reward"], "sinr_improvement": avg_pre_metrics["sinr_violations"] - avg_post_metrics["sinr_violations"], "latency_improvement": avg_pre_metrics["latency_violations"] - avg_post_metrics["latency_violations"],
            }
            all_metrics.append(eval_summary)

            print(f"[Iter {i:04d} EVAL] Pre-Reward: {eval_summary['avg_pre_reward']:.3f}, Post-Reward: {eval_summary['avg_post_reward']:.3f} "
                  f"(Impr: {eval_summary['reward_improvement']:.3f})")
            print(f"               Pre-SINR Viol.: {eval_summary['avg_pre_sinr_violations']:.3f}, Post-SINR Viol.: {eval_summary['avg_post_sinr_violations']:.3f}")
            
            if h_config.use_wandb:
                wandb.log({"iteration": i, **eval_summary})

    if h_config.use_wandb:
        wandb.finish()
    
    return params, all_metrics

if __name__ == '__main__':
    config = {
        "seed": 42, "meta_lr": 3e-5, "inner_lr": 1e-4, "inner_steps": 5, "meta_batch_size": 16,
        "num_meta_iters": 500, "rollout_length": 128, "vf_coef": 0.5, "ent_coef": 0.01,
        "clip_epsilon": 0.2, "max_grad_norm": 1.0, "use_wandb": False, "gamma": 0.99, "lambda_": 0.95,
        "wandb_project": "maml-spectrum-access", "wandb_name": "simple-recurrent-maml-ppo",
        
        # Evaluation and Logging
        "log_interval": 10,
        "eval_interval": 50,
        "num_eval_tasks": 10,

        # Network Hyperparameters
        "lstm_hidden_dim": 64,
        "mlp_hidden_dim": 128,
        "critic_hidden_dim": 128,
    }
    print("Starting Simple Recurrent MAML-PPO with detailed evaluation")
    trained_params, metrics_history = train_recurrent_maml_ppo(config)
    print("Training finished.")