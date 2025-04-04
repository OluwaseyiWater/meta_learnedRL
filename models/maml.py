from jumanji.types import TimeStep
from functools import partial
from typing import List, Dict, Any
import haiku as hk
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax as tfp

# Constants (assumed to be defined elsewhere or provided by DynamicSpectrumEnv)
NUM_BS = 3  
NUM_BANDS = 4  
NUM_USERS = 5  
NUM_POWER_LEVELS = 5  

# Hyperparameters
META_LR = 1e-3
INNER_LR = 0.1
META_BATCH_SIZE = 4
NUM_INNER_STEPS = 1
NUM_META_ITERS = 1000
ROLLOUT_LENGTH = 50
DISCOUNT_FACTOR = 0.99  
NUM_META_BATCHES = 10



tfd = tfp.distributions

# Define the policy network
class MLPNetwork(hk.Module):
    def __init__(self, num_bs, num_bands, num_power_levels):
        super().__init__()
        self.num_bs = num_bs
        self.num_bands = num_bands
        self.num_power_levels = num_power_levels

    def __call__(self, x):
        x = hk.Flatten()(x)
        x = hk.Linear(64)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(64)(x)
        x = jax.nn.relu(x)
        # Output logits for each BS-band pair's power level choices
        logits = hk.Linear(self.num_bs * self.num_bands * self.num_power_levels)(x)
        logits = logits.reshape(-1, self.num_bs * self.num_bands, self.num_power_levels)
        return logits

# Policy function
def policy_fn(obs, num_bs, num_bands, num_power_levels):
    logits = MLPNetwork(num_bs, num_bands, num_power_levels)(obs)
    return tfd.Categorical(logits=logits)

# Transform the policy for Haiku
def make_policy(num_bs, num_bands, num_power_levels):
    return hk.without_apply_rng(hk.transform(
        lambda obs: policy_fn(obs, num_bs, num_bands, num_power_levels)
    ))

def flatten_state(state):
    return jnp.concatenate([
        state.channel_gains.flatten(),
        state.interference_map.flatten(),
        state.qos_metrics.flatten(),
        state.spectrum_alloc.flatten().astype(jnp.float32),
        state.tx_power.flatten()
    ])

# Sample trajectories from the environment
def sample_trajectories(env, params, key, max_steps=100):
    state, _ = env.reset(key)
    obs = flatten_state(state)
    observations, actions, rewards, dones = [], [], [], []
    
    for _ in range(max_steps):
        key, subkey = jax.random.split(key)
        action_dist = policy.apply(params, obs)
        action = action_dist.sample(seed=subkey)
        next_state, timestep = env.step(state, action)
        next_obs = flatten_state(next_state)
        
        observations.append(obs)
        actions.append(action)
        rewards.append(timestep.reward)
        dones.append(timestep.last())
        
        obs = next_obs
        state = next_state
        if timestep.last():
            break
    
    return {
        'observations': jnp.stack(observations),
        'actions': jnp.stack(actions),
        'rewards': jnp.stack(rewards),
        'dones': jnp.stack(dones)
    }

# Compute advantages (simplified, assumes a baseline or value function could be added)
def compute_advantages(traj):
    rewards = traj['rewards']
    # Simple cumulative reward as advantage (extend with GAE or value function if needed)
    returns = jnp.cumsum(rewards[::-1])[::-1]
    return returns

# Compute policy loss and gradients
def compute_policy_loss(params, traj):
    obs = traj['observations']
    actions = traj['actions']
    advantages = compute_advantages(traj)
    advantages = jnp.expand_dims(advantages, axis=(1, 2))
    
    
    action_dist = policy.apply(params, obs)
    log_probs = action_dist.log_prob(actions)
    loss = -jnp.mean(log_probs * advantages)
    return loss

# MAML training loop
def train_maml(env, num_meta_tasks=100, inner_steps=1, meta_lr=1e-3, inner_lr=1e-2):
    # Extract environment parameters
    num_bs = env.num_bs
    num_bands = env.num_bands
    num_power_levels = env.num_power_levels
    
    # Initialize policy
    global policy  
    policy = make_policy(num_bs, num_bands, num_power_levels)
    key = jax.random.PRNGKey(42)
    state, _ = env.reset(key)
    sample_obs = flatten_state(state)
    params = policy.init(key, sample_obs)
    
    # Optimizers
    meta_opt = optax.adam(meta_lr)
    meta_opt_state = meta_opt.init(params)
    inner_opt = optax.sgd(inner_lr)
    
    # Meta-training loop
    for meta_task in range(num_meta_tasks):
        key, task_key = jax.random.split(key)
        
        # Sample a task (assuming reset can vary scenarios)
        traj = sample_trajectories(env, params, task_key)
        
        # Inner loop: adapt to the task
        inner_params = params
        for _ in range(inner_steps):
            loss, grads = jax.value_and_grad(compute_policy_loss)(inner_params, traj)
            updates, _ = inner_opt.update(grads, inner_opt.init(inner_params))
            inner_params = optax.apply_updates(inner_params, updates)
        
        # Outer loop: compute meta-gradient
        meta_loss, meta_grads = jax.value_and_grad(lambda p: compute_policy_loss(p, traj))(params)
        updates, meta_opt_state = meta_opt.update(meta_grads, meta_opt_state)
        params = optax.apply_updates(params, updates)
        
        if meta_task % 10 == 0:
            print(f"Meta-task {meta_task}, Meta Loss: {meta_loss:.4f}")

    return params
