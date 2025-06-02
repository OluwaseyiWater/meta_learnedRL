import jax
import jax.numpy as jnp
import jax.random as jr
import chex
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep, restart, termination, transition, truncation
from typing import Optional, Tuple, Dict
import numpy as np
import optax 

# Constants and hyperparameters
MAX_INTERFERENCE = 25.0  
MAX_POWER = 23.0  
MAX_LATENCY = 50.0  
NUM_BS = 3
NUM_USERS = 10
NUM_BANDS = 5
NUM_POWER_LEVELS = 4 
POWER_LEVELS = jnp.linspace(0, MAX_POWER, NUM_POWER_LEVELS) 
FADING_COHERENCE = 0.9  
MAX_STEPS = 100 
MIN_SINR = 5.0 

@chex.dataclass
class SpectrumState:
    channel_gains: chex.Array      
    interference_map: chex.Array   
    qos_metrics: chex.Array        
    spectrum_alloc: chex.Array     
    tx_power: chex.Array           
    time: chex.Array               
    key: chex.PRNGKey             

class DynamicSpectrumEnv(Environment):
    def __init__(self, num_bs=NUM_BS, num_users=NUM_USERS, num_bands=NUM_BANDS, max_steps=MAX_STEPS,
                 max_latency=MAX_LATENCY, max_power=MAX_POWER, num_power_levels=NUM_POWER_LEVELS, power_levels=POWER_LEVELS,
                 fading_coherence=FADING_COHERENCE, max_interference=MAX_INTERFERENCE, min_sinr=MIN_SINR):
        self.num_bs = num_bs
        self.num_users = num_users
        self.num_bands = num_bands
        self.max_steps = max_steps
        self.max_latency = max_latency
        self.max_power = max_power
        self.num_power_levels = num_power_levels
        self.power_levels = power_levels
        self.fading_coherence = fading_coherence
        self.max_interference = max_interference
        self.min_sinr = min_sinr

    def action_spec(self) -> specs.MultiDiscreteArray:
        return specs.MultiDiscreteArray(
            num_values=jnp.full((self.num_bs * self.num_bands,), self.num_power_levels, dtype=jnp.int32),
            name='action'
        )

    def observation_spec(self) -> specs.Spec:
        return specs.Spec(
            SpectrumState,
            "Observation",
            channel_gains=specs.Array((self.num_users, self.num_bs), jnp.float32),
            interference_map=specs.Array((self.num_bs, self.num_bands), jnp.float32),
            qos_metrics=specs.Array((self.num_users, 2), jnp.float32),
            spectrum_alloc=specs.Array((self.num_bs, self.num_bands), jnp.int32),
            tx_power=specs.Array((self.num_bs, self.num_bands), jnp.float32),
            time=specs.Array((), jnp.int32),
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[SpectrumState, TimeStep]:
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        scenario = jax.random.randint(subkey1, (), 0, 3)
        path_loss = jnp.select(
            [scenario == 0, scenario == 1, scenario == 2],
            [128.1 + 37.6 * jnp.log10(0.5), 98.5 + 23.1 * jnp.log10(2.0), 105.3 + 34.2 * jnp.log10(1.0)]
        )
        channel_gains = path_loss + jax.random.normal(subkey2, (self.num_users, self.num_bs))
        interference_map = jax.random.uniform(subkey3, (self.num_bs, self.num_bands), minval=0.0, maxval=self.max_interference * 0.5)
        state = SpectrumState(
            channel_gains=channel_gains,
            interference_map=interference_map,
            qos_metrics=jnp.zeros((self.num_users, 2), dtype=jnp.float32),
            spectrum_alloc=jnp.zeros((self.num_bs, self.num_bands), dtype=jnp.int32),
            tx_power=jnp.zeros((self.num_bs, self.num_bands), dtype=jnp.float32),
            time=jnp.array(0, dtype=jnp.int32),
            key=key
        )
        return state, restart(state)

    def _mask_unsafe_actions(self, state: SpectrumState) -> chex.Array:
        interference_mask = state.interference_map[:, :, None] < self.max_interference
        return interference_mask

    def _compute_sinr(self, state: SpectrumState) -> jnp.ndarray:
        avg_channel_gain = jnp.mean(state.channel_gains, axis=0)[:, None]
        signal = state.tx_power - jnp.tile(avg_channel_gain, (1, self.num_bands))
        interference_db = 10 * jnp.log10(state.interference_map + 1e-6)
        sinr = signal - interference_db
        return sinr

    def _calculate_reward(self, state: SpectrumState, action: jnp.ndarray, previous_tx_power: jnp.ndarray) -> float:
        POWER_COST_COEFF = 0.1
        SWITCHING_COST_COEFF = 1.0
        UTILIZATION_BONUS_COEFF = 2.0
        VIOLATION_PENALTY_COEFF = 10.0
        FAIRNESS_COEFF = 0.5
        # Calculate reward components
        throughput = jnp.sum(state.qos_metrics[:, 1])
        x = state.qos_metrics[:,1]
        fairness = (jnp.sum(x)**2) / (x.shape[0] * (jnp.sum(x**2) + 1e-6))
        sinr = self._compute_sinr(state)
        sinr_violations = jnp.sum(sinr < self.min_sinr)
        latency_violations = jnp.sum(state.qos_metrics[:, 0] > self.max_latency)
        violation_penalty = VIOLATION_PENALTY_COEFF * (latency_violations + sinr_violations)
        new_tx_power = self.power_levels[action]
        power_cost = POWER_COST_COEFF * jnp.sum(new_tx_power)
        switching_cost = SWITCHING_COST_COEFF * jnp.sum(jnp.abs(previous_tx_power - new_tx_power))
        utilization_bonus = UTILIZATION_BONUS_COEFF * jnp.sum(action > 0)
        total_reward = throughput + FAIRNESS_COEFF * fairness + utilization_bonus - violation_penalty - power_cost - switching_cost
        return total_reward

    def _adaptive_penalty(self, state: SpectrumState) -> jnp.ndarray:
        penalty = 0.1 * jnp.sum(state.interference_map > (0.8 * self.max_interference))
        return penalty

    def _step_dynamics(self, state: SpectrumState, action: jnp.ndarray) -> SpectrumState:
        key, subkey = jax.random.split(state.key)
        new_channel = state.channel_gains * self.fading_coherence + jax.random.normal(subkey, state.channel_gains.shape)
        new_alloc = action
        new_tx_power = self.power_levels[action]
        new_qos = state.qos_metrics.at[:, 0].add(1.0)
        total_power = jnp.sum(new_tx_power)
        throughput_bonus = total_power / (self.num_bs * self.num_bands * self.max_power)
        new_throughput = state.qos_metrics[:, 1] + throughput_bonus
        new_qos = new_qos.at[:, 1].set(new_throughput)
        return SpectrumState(
            channel_gains=new_channel,
            interference_map=state.interference_map,
            qos_metrics=new_qos,
            spectrum_alloc=new_alloc,
            tx_power=new_tx_power,
            time=state.time + 1,
            key=key
        )

    
    def step(self, state: SpectrumState, action: chex.Array) -> Tuple[SpectrumState, TimeStep]:
        action = action.reshape(self.num_bs, self.num_bands)
        action_mask = self._mask_unsafe_actions(state)
        safety = action_mask[jnp.arange(self.num_bs)[:, None], jnp.arange(self.num_bands)[None, :], action]
        safe_action = jnp.where(safety, action, 0)
        previous_tx_power = state.tx_power
        new_state = self._step_dynamics(state, safe_action)
        reward = self._calculate_reward(new_state, safe_action, previous_tx_power) - self._adaptive_penalty(new_state)
        
       
        terminated = jnp.any(new_state.qos_metrics[:, 0] > 2 * self.max_latency)
        truncated = new_state.time >= self.max_steps
        done_flag = jnp.logical_or(terminated, truncated)
        

        timestep = jax.lax.cond(
            done_flag,
            lambda: jax.lax.cond(
                terminated,  
                lambda: termination(reward, new_state),
                lambda: truncation(reward, new_state),
            ),
            lambda: transition(reward, new_state),
        )
        return new_state, timestep

    def render(self, state: SpectrumState) -> None:
        print(f"Step {int(state.time)}")
        print("Spectrum Allocation:")
        print(state.spectrum_alloc)
        print("Transmit Power:")
        print(state.tx_power)
        print(f"Latency violations: {jnp.sum(state.qos_metrics[:, 0] > self.max_latency)}")
        print(f"SINR violations: {jnp.sum(self._compute_sinr(state) < self.min_sinr)}")
        
# Test and Usage
if __name__ == "__main__":
    from jax import random
    import jax.numpy as jnp
    
    # Initialize environment
    env = DynamicSpectrumEnv()
    key = random.PRNGKey(42)
    state, timestep = env.reset(key)
    
    action = jnp.ones((env.num_bs * env.num_bands,), dtype=jnp.int32)
    new_state, timestep = env.step(state, action)
    
    action_mask = env._mask_unsafe_actions(new_state)
    print(f"Action mask for power level 3 on BS 0, band 0: {action_mask[0, 0, 3]}")
    
    def train(num_episodes=100):
        params = {}  
        opt = optax.adam(1e-3)
        opt_state = opt.init(params)
        
        for episode in range(num_episodes):
            key = random.PRNGKey(episode)
            state, _ = env.reset(key)
            done = False
            while not done:
                state.key, subkey = jax.random.split(state.key)
                action = jax.random.randint(subkey, (env.num_bs * env.num_bands,), minval=0, maxval=env.num_power_levels)
                state, timestep = env.step(state, action)
                done = bool(timestep.last())
                
            print(f"Episode {episode} completed at step {int(state.time)}")
    
   
    train(num_episodes=3)
