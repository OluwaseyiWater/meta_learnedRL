import jax
import jax.numpy as jnp
import optax
import haiku as hk
import chex
import wandb 
from functools import partial
from typing import Any, Callable, Tuple, Dict, Optional, Sequence
from collections import namedtuple
import tensorflow_probability.substrates.jax as tfp


from mLN.environment import DynamicSpectrumEnv, SpectrumState 
def flatten_state(state: SpectrumState) -> jnp.ndarray:
    return jnp.concatenate([
        state.channel_gains.flatten(),
        state.interference_map.flatten(),
        state.user_latency.flatten(),
        state.spectrum_alloc.flatten(),
        state.tx_power.flatten(),
        jnp.array([state.time]),
    ])


tfd = tfp.distributions
class RecurrentActor(hk.Module):
    def __init__(self, lstm_hidden_dim: int, mlp_hidden_dim: int, num_outputs: int):
        super().__init__()
        self.lstm = hk.LSTM(lstm_hidden_dim)
        self.body = hk.nets.MLP([mlp_hidden_dim, mlp_hidden_dim])
        self.head = hk.Linear(num_outputs)

    def __call__(self, obs: chex.Array, state: Optional[hk.LSTMState]) -> Tuple[chex.Array, hk.LSTMState]:
        batch_size = obs.shape[0]
        if state is None:
            state = self.lstm.initial_state(batch_size)
        
        embedding, new_state = self.lstm(obs, state)
        logits = self.head(self.body(embedding))
        return jnp.clip(logits, -10.0, 10.0), new_state

class Critic(hk.Module):
    def __init__(self, hidden_dim=128): 
        super().__init__()
        self.body = hk.nets.MLP([hidden_dim, hidden_dim])
        self.head = hk.Linear(1)

    def __call__(self, obs: chex.Array) -> chex.Array:
        return self.head(self.body(hk.Flatten()(obs))).squeeze(axis=-1)

def make_recurrent_networks_ppo(num_outputs: int, config: Any):
    def actor_fn(obs: chex.Array, state: Optional[hk.LSTMState]):
        net = RecurrentActor(
            lstm_hidden_dim=config.lstm_hidden_dim,
            mlp_hidden_dim=config.mlp_hidden_dim,
            num_outputs=num_outputs
        )
        return net(obs, state)

    def critic_fn(obs: chex.Array):
        net = Critic(hidden_dim=config.critic_hidden_dim)
        return net(obs)

    actor = hk.without_apply_rng(hk.transform(actor_fn))
    critic = hk.without_apply_rng(hk.transform(critic_fn))
    return actor, critic

# 2. Observation Normalization 
ObsNormalizerState = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] 

def init_obs_normalizer(obs_dim: int) -> ObsNormalizerState:
    return (jnp.zeros(obs_dim), jnp.ones(obs_dim), jnp.array(1e-8)) 

def update_obs_normalizer(state: ObsNormalizerState, obs_batch: jnp.ndarray) -> ObsNormalizerState:
    mean, var, count = state
    
    # Welford's algorithm for online variance
    batch_count = obs_batch.shape[0]
    if batch_count == 0:
        return state

    delta = obs_batch - mean # Broadcasting mean
    new_count = count + batch_count
    
    # Update mean
    batch_mean = jnp.mean(obs_batch, axis=0)
    delta_mean = batch_mean - mean
    new_mean = mean + delta_mean * (batch_count / new_count)

    # Update variance (M2)
    new_var_sum = var * count + jnp.sum(jnp.square(obs_batch - new_mean) + jnp.square(mean - new_mean), axis=0) \
                  + 2 * jnp.sum((obs_batch - new_mean)*(new_mean - mean), axis=0) 
    
    
    if count == 1e-8: 
        new_var = jnp.var(obs_batch, axis=0)
    else:
        m_a = var * count 
        m_b = jnp.var(obs_batch, axis=0) * batch_count
        M2 = m_a + m_b + jnp.square(delta_mean) * count * batch_count / new_count
        new_var = M2 / new_count
        
    return (new_mean, jnp.maximum(new_var, 1e-8), new_count)


def normalize_obs(norm_state: ObsNormalizerState, obs: jnp.ndarray) -> jnp.ndarray:
    mean, var, _ = norm_state
    return (obs - mean) / jnp.sqrt(var + 1e-8) 

#  Trajectory Sampling 
def _get_env_metrics(state: SpectrumState, env: DynamicSpectrumEnv) -> Dict[str, chex.Array]:
    """ Helper to compute environment-specific metrics from a state."""
    metrics = env._compute_metrics(state) 
    return {
        "sinr_violations": metrics["sinr_violations"],
        "latency_violations": metrics["latency_violations"],
        "total_throughput": metrics["total_throughput"],
        "fairness_index": metrics["fairness_index"]
    }

def sample_trajectories_ppo(
    env: DynamicSpectrumEnv,
    actor_params: Any,
    critic_params: Any,
    actor_apply_fn: Callable,
    critic_apply_fn: Callable,
    key: jax.random.PRNGKey,
    obs_norm_state: ObsNormalizerState,
    rollout_length: int,
    num_actions: int, 
    num_action_values: int 
) -> Dict[str, jnp.ndarray]:
    
    env_state, timestep = env.reset(key)
    actor_lstm_state = actor_apply_fn(actor_params, normalize_obs(obs_norm_state, flatten_state(env_state))[None, :], None)[1]

    obs_list, action_list, reward_list, value_list, done_list, log_prob_list = [], [], [], [], [], []
    sinr_violations_list, latency_violations_list, throughput_list, fairness_list = [], [], [], []
    
    current_obs_norm_state = obs_norm_state 
    for t in range(rollout_length):
        obs_flat = flatten_state(env_state)
        obs_list.append(obs_flat) 

        norm_obs = normalize_obs(current_obs_norm_state, obs_flat)[None, :] 

        logits, new_actor_lstm_state = actor_apply_fn(actor_params, norm_obs, actor_lstm_state)
        action_dist = tfd.Categorical(logits=logits.reshape(1, num_actions, num_action_values))
        
        key, action_subkey = jax.random.split(key)
        action = action_dist.sample(seed=action_subkey) 
        log_prob = action_dist.log_prob(action)       

        
        action_env = action.squeeze(0).reshape(-1) 
        
        value = critic_apply_fn(critic_params, norm_obs.squeeze(0)) 

        next_env_state, timestep = env.step(env_state, action_env)
        env_metrics = _get_env_metrics(next_env_state, env)

        action_list.append(action.squeeze(0))
        reward_list.append(timestep.reward)
        value_list.append(value)
        done_list.append(timestep.last())
        log_prob_list.append(log_prob.squeeze(0)) 

        sinr_violations_list.append(env_metrics["sinr_violations"])
        latency_violations_list.append(env_metrics["latency_violations"])
        throughput_list.append(env_metrics["total_throughput"])
        fairness_list.append(env_metrics["fairness_index"])

        env_state = next_env_state
        actor_lstm_state = new_actor_lstm_state
        
        if timestep.last():
            pass 

    final_obs_flat = flatten_state(env_state)
    norm_final_obs = normalize_obs(current_obs_norm_state, final_obs_flat)
    final_value = critic_apply_fn(critic_params, norm_final_obs)

    trajectory = {
        "obs": jnp.array(obs_list),
        "actions": jnp.array(action_list),
        "rewards": jnp.array(reward_list),
        "values": jnp.array(value_list),
        "dones": jnp.array(done_list),
        "log_probs": jnp.array(log_prob_list), 
        "final_value": final_value,
        "sinr_violations": jnp.array(sinr_violations_list),
        "latency_violations": jnp.array(latency_violations_list),
        "total_throughput": jnp.array(throughput_list),
        "fairness_index": jnp.array(fairness_list),
    }
    return trajectory


#  GAE and PPO Loss 
def compute_gae_ppo(traj: Dict, gamma: float, lambda_: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    last_value = traj["final_value"]
    def gae_scan_fn(carry, transition):
        gae_plus_1, value_plus_1 = carry
        reward_t, done_t, value_t = transition 
        next_val_if_not_done = value_plus_1 
        true_next_value = jnp.where(done_t, 0.0, next_val_if_not_done)

        delta_t = reward_t + gamma * true_next_value - value_t
        gae_t = delta_t + gamma * lambda_ * jnp.where(done_t, 0.0, gae_plus_1) 
        return (gae_t, value_t), gae_t 

    values_for_scan = traj["values"] # V(s_0) to V(s_{T-1})
    final_v = jnp.where(traj["dones"][-1], 0.0, traj["final_value"]) # V(s_T) is 0 if s_{T-1} was terminal leading to s_T as the 'next' state
                                                                 
    v_at_T = jnp.where(traj["dones"][-1], 0.0, traj["final_value"])


    _, advantages_reversed = jax.lax.scan(
        gae_scan_fn,
        (0.0, v_at_T), 
        (traj["rewards"], traj["dones"], traj["values"]), 
        reverse=True
    )
    advantages = advantages_reversed 
    
    returns = advantages + traj["values"] 
    return jax.lax.stop_gradient(returns), jax.lax.stop_gradient(advantages)


def compute_ppo_loss_fn(
    actor_params: Any,
    critic_params: Any, 
    actor_apply_fn: Callable,
    
    batch: Dict, 
    obs_norm_state: ObsNormalizerState, 
    config: Any, 
    num_actions: int,
    num_action_values: int
) -> Tuple[jnp.ndarray, Dict]:
    
    norm_obs_batch = jax.vmap(normalize_obs, in_axes=(None, 0))(obs_norm_state, batch['obs'])
    
    _, initial_actor_lstm_state_for_loss = actor_apply_fn(actor_params, norm_obs_batch[0][None,:], None)
    
    def policy_eval_scan_fn(carry_lstm_state, obs_t_norm):
        logits_t, next_lstm_state = actor_apply_fn(actor_params, obs_t_norm[None, :], carry_lstm_state)
        dist_t = tfd.Categorical(logits=logits_t.reshape(1, num_actions, num_action_values))
        return next_lstm_state, (dist_t, logits_t)

    
    final_lstm_state, (action_dists, _) = jax.lax.scan(
        policy_eval_scan_fn,
        initial_actor_lstm_state_for_loss,
        norm_obs_batch 
    )
    

    def get_logprobs_entropy(single_norm_obs, single_action, h_state):
        logits, new_h_state = actor_apply_fn(actor_params, single_norm_obs[None,:], h_state)
        dist = tfd.Categorical(logits=logits.reshape(1, num_actions, num_action_values))
        return new_h_state, (dist.log_prob(single_action[None,:]).squeeze(0), dist.entropy().squeeze(0))

    
    
    initial_logprob_h_state = actor_apply_fn(actor_params, norm_obs_batch[0][None,:], None)[1] 
    shifted_dones = jnp.insert(batch['dones'][:-1], 0, True) 
    def scan_body_for_loss(h_state, inputs):
        obs_t, action_t, done_prev_t = inputs 
        
        current_h_state = jax.lax.cond(
            done_prev_t,
            lambda _: actor_apply_fn(actor_params, obs_t[None,:], None)[1], 
            lambda _: h_state,
            operand=None
        )
        
        logits, next_h_state = actor_apply_fn(actor_params, obs_t[None,:], current_h_state)
        dist = tfd.Categorical(logits=logits.reshape(1, num_actions, num_action_values))
        log_prob_t = dist.log_prob(action_t[None,:]).squeeze(0) 
        entropy_t = dist.entropy().squeeze(0) 
        return next_h_state, (log_prob_t, entropy_t)

    _, (new_log_probs_per_component, new_entropies_per_component) = jax.lax.scan(
        scan_body_for_loss,
        initial_logprob_h_state,
        (norm_obs_batch, batch['actions'], shifted_dones)
    )

    summed_new_log_probs = jnp.sum(new_log_probs_per_component, axis=1) 
    summed_old_log_probs = jax.lax.stop_gradient(jnp.sum(batch['log_probs'], axis=1))

    ratio = jnp.exp(summed_new_log_probs - summed_old_log_probs)
    advantages = batch['advantages'] 
    
    # Normalize advantages 
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    policy_loss_term1 = ratio * advantages
    policy_loss_term2 = jnp.clip(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * advantages
    actor_loss = -jnp.mean(jnp.minimum(policy_loss_term1, policy_loss_term2))

    # Entropy loss
    entropy_loss = -jnp.mean(jnp.sum(new_entropies_per_component, axis=1))
    
    total_actor_loss = actor_loss + config.ent_coef * entropy_loss
    
    metrics = {
        "actor_loss": actor_loss,
        "entropy_loss": entropy_loss,
        "total_actor_loss": total_actor_loss,
        "ppo_ratio": jnp.mean(ratio)
    }
    return total_actor_loss, metrics

def compute_critic_loss_fn(
    critic_params: Any,
    critic_apply_fn: Callable,
    batch: Dict, 
    obs_norm_state: ObsNormalizerState, 
    config: Any
) -> Tuple[jnp.ndarray, Dict]:

    norm_obs_batch = jax.vmap(normalize_obs, in_axes=(None, 0))(obs_norm_state, batch['obs'])
    
    
    values_pred = jax.vmap(critic_apply_fn, in_axes=(None, 0))(critic_params, norm_obs_batch)
    
    returns = batch['returns'] 
    
    value_loss = jnp.mean(optax.huber_loss(values_pred, jax.lax.stop_gradient(returns)))
    metrics = {"value_loss": value_loss}
    return value_loss, metrics


#  PPO Update Step 
@partial(jax.jit, static_argnames=(
    "actor_apply_fn", "critic_apply_fn", "actor_optimizer", "critic_optimizer",
    "config", "num_actions", "num_action_values"
))
def ppo_update_step(
    actor_params: Any,
    critic_params: Any,
    actor_opt_state: Any,
    critic_opt_state: Any,
    batch: Dict, 
    obs_norm_state: ObsNormalizerState, 
    actor_apply_fn: Callable,
    critic_apply_fn: Callable,
    actor_optimizer: optax.GradientTransformation,
    critic_optimizer: optax.GradientTransformation,
    config: Any, 
    num_actions: int,
    num_action_values: int
) -> Tuple[Any, Any, Any, Any, Dict]:

    # Actor update
    grad_actor_fn = jax.value_and_grad(compute_ppo_loss_fn, argnums=0, has_aux=True)
    (total_actor_loss, actor_metrics), actor_grads = grad_actor_fn(
        actor_params, critic_params, 
        actor_apply_fn, batch, obs_norm_state, config, num_actions, num_action_values
    )
    actor_updates, new_actor_opt_state = actor_optimizer.update(actor_grads, actor_opt_state, actor_params)
    new_actor_params = optax.apply_updates(actor_params, actor_updates)

    # Critic update
    grad_critic_fn = jax.value_and_grad(compute_critic_loss_fn, argnums=0, has_aux=True)
    (value_loss, critic_metrics), critic_grads = grad_critic_fn(
        critic_params, critic_apply_fn, batch, obs_norm_state, config
    )
    critic_updates, new_critic_opt_state = critic_optimizer.update(critic_grads, critic_opt_state, critic_params)
    new_critic_params = optax.apply_updates(critic_params, critic_updates)
    
    all_metrics = {**actor_metrics, **critic_metrics}
    all_metrics["mean_reward_in_batch"] = jnp.mean(batch["rewards"])
    all_metrics["mean_sinr_violations_in_batch"] = jnp.mean(batch["sinr_violations"])
    all_metrics["mean_latency_violations_in_batch"] = jnp.mean(batch["latency_violations"])
    all_metrics["mean_total_throughput_in_batch"] = jnp.mean(batch["total_throughput"])
    all_metrics["mean_fairness_index_in_batch"] = jnp.mean(batch["fairness_index"])


    return new_actor_params, new_critic_params, new_actor_opt_state, new_critic_opt_state, all_metrics

if __name__ == '__main__':

    print("PPO JAX implementation structure defined.")
    print("A full training script would use these components.")

    DummyPPOConfig = namedtuple("DummyPPOConfig", [
        "lstm_hidden_dim", "mlp_hidden_dim", "critic_hidden_dim", "clip_epsilon", "ent_coef",
        "gamma", "lambda_", "learning_rate_actor", "learning_rate_critic", "max_grad_norm"
    ])
    dummy_config = DummyPPOConfig(
        lstm_hidden_dim=64, mlp_hidden_dim=128, critic_hidden_dim=128,
        clip_epsilon=0.2, ent_coef=0.01, gamma=0.99, lambda_=0.95,
        learning_rate_actor=3e-4, learning_rate_critic=3e-4, max_grad_norm=0.5
    )

    env = DynamicSpectrumEnv()
    num_total_actions = env.num_bs * env.num_bands 
    num_power_options = env.num_power_levels      
    actor_output_size = num_total_actions * num_power_options

    actor_net, critic_net = make_recurrent_networks_ppo(actor_output_size, dummy_config)

    key = jax.random.PRNGKey(0)
    key_actor, key_critic, key_rollout, key_update = jax.random.split(key, 4)

    # Initialize networks
    dummy_flat_obs = flatten_state(env.observation_spec().generate_value())
    dummy_actor_input_obs = dummy_flat_obs[None, :] # Add batch dim
    
    actor_params = actor_net.init(key_actor, dummy_actor_input_obs, None) 
    critic_params = critic_net.init(key_critic, dummy_flat_obs) 

    # Initialize Obs Normalizer
    obs_norm_state = init_obs_normalizer(dummy_flat_obs.shape[0])

    print(f"Actor output size: {actor_output_size}")
    print(f"Num independent actions (e.g. BS-Band pairs): {num_total_actions}")
    print(f"Num power levels per action: {num_power_options}")


    # Test trajectory sampling
    print("\nTesting trajectory sampling...")
    try:
        trajectory_data = sample_trajectories_ppo(
            env, actor_params, critic_params, actor_net.apply, critic_net.apply,
            key_rollout, obs_norm_state, rollout_length=50,
            num_actions=num_total_actions, num_action_values=num_power_options
        )
        print("Trajectory sampling successful. Keys:", trajectory_data.keys())
        print("Obs shape:", trajectory_data["obs"].shape)
        print("Actions shape:", trajectory_data["actions"].shape)
        print("Log_probs shape:", trajectory_data["log_probs"].shape) 

       
        obs_norm_state = update_obs_normalizer(obs_norm_state, trajectory_data["obs"])
        print("Obs normalizer updated.")

        # Compute GAE
        returns, advantages = compute_gae_ppo(trajectory_data, dummy_config.gamma, dummy_config.lambda_)
        trajectory_data["returns"] = returns
        trajectory_data["advantages"] = advantages
        print("GAE computation successful.")
        print("Returns shape:", trajectory_data["returns"].shape)
        print("Advantages shape:", trajectory_data["advantages"].shape)


        # Test PPO loss and update (dummy optimizers for test)
        actor_optimizer = optax.adam(dummy_config.learning_rate_actor)
        critic_optimizer = optax.adam(dummy_config.learning_rate_critic)
        actor_opt_state = actor_optimizer.init(actor_params)
        critic_opt_state = critic_optimizer.init(critic_params)
        
       
        print("\nTesting PPO update step...")
        new_actor_params, new_critic_params, _, _, metrics = ppo_update_step(
            actor_params, critic_params, actor_opt_state, critic_opt_state,
            trajectory_data, 
            obs_norm_state,  
            actor_net.apply, critic_net.apply,
            actor_optimizer, critic_optimizer,
            dummy_config, num_total_actions, num_power_options
        )
        print("PPO update step successful. Metrics:", metrics)

    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
