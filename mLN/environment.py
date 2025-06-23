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


# Environment Constants
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

# ==============================================================================
# CONSTANTS
# ==============================================================================
ROLLOUT_LENGTH = 50 
BANDWIDTH_HZ = 10e6
NOISE_FIGURE_DB = 7.0

@chex.dataclass
class SpectrumState:
    channel_gains: chex.Array
    interference_map: chex.Array
    user_latency: chex.Array
    spectrum_alloc: chex.Array
    tx_power: chex.Array
    time: chex.Array
    key: Optional[chex.PRNGKey] = None

class DynamicSpectrumEnv(Environment):
    def __init__(self, num_bs=NUM_BS, num_users=NUM_USERS, num_bands=NUM_BANDS, max_steps=MAX_STEPS,
                 max_latency=MAX_LATENCY, num_power_levels=NUM_POWER_LEVELS, power_levels=POWER_LEVELS,
                 fading_coherence=FADING_COHERENCE, max_interference=MAX_INTERFERENCE, min_sinr=MIN_SINR):
        self.num_bs = num_bs
        self.num_users = num_users
        self.num_bands = num_bands
        self.max_steps = max_steps
        self.max_latency = max_latency
        self.num_power_levels = num_power_levels
        self.power_levels = power_levels
        self.fading_coherence = fading_coherence
        self.max_interference = max_interference
        self.min_sinr = min_sinr
        # Additional parameters for proper wireless modeling
        self.noise_figure_db = NOISE_FIGURE_DB
        self.bandwidth_hz = BANDWIDTH_HZ
        self.thermal_noise_dbm_hz = THERMAL_NOISE_DBM_HZ

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
            user_latency=specs.Array((self.num_users,), jnp.float32),
            spectrum_alloc=specs.Array((self.num_bs, self.num_bands), jnp.int32),
            tx_power=specs.Array((self.num_bs, self.num_bands), jnp.float32),
            time=specs.Array((), jnp.int32),
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[SpectrumState, TimeStep]:
        key, subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 5)
        
        
        scenario = jax.random.randint(subkey1, (), 0, 3)
        # Distance-dependent path loss (in dB)
        distances = jax.random.uniform(subkey4, (self.num_users, self.num_bs), minval=0.1, maxval=2.0)  # km
        
        # Path loss models for different scenarios (Urban, Suburban, Rural)
        path_loss_urban = 128.1 + 37.6 * jnp.log10(distances)
        path_loss_suburban = 98.5 + 23.1 * jnp.log10(distances) 
        path_loss_rural = 105.3 + 34.2 * jnp.log10(distances)
        
        path_loss = jnp.select(
            [scenario == 0, scenario == 1, scenario == 2],
            [path_loss_urban, path_loss_suburban, path_loss_rural]
        )
        
        #  shadow fading (log-normal)
        shadow_fading = jax.random.normal(subkey2, (self.num_users, self.num_bs)) * 8.0  # 8 dB std
        
        # Total channel gains
        channel_gains = -(path_loss + shadow_fading)
        
        # External interference 
        interference_mw = jax.random.uniform(subkey3, (self.num_bs, self.num_bands), 
                                           minval=0.001, maxval=self.max_interference * 0.5)
        interference_map = 10.0 * jnp.log10(interference_mw + 1e-12)  
        
        state = SpectrumState(
            channel_gains=channel_gains,
            interference_map=interference_map,
            user_latency=jnp.zeros(self.num_users, dtype=jnp.float32),
            spectrum_alloc=jnp.zeros((self.num_bs, self.num_bands), dtype=jnp.int32),
            tx_power=jnp.zeros((self.num_bs, self.num_bands), dtype=jnp.float32),
            time=jnp.array(0, dtype=jnp.int32),
            key=key
        )
        return state, restart(state)

    def _mask_unsafe_actions(self, state: SpectrumState) -> chex.Array:
        return state.interference_map < self.max_interference

    def _compute_sinr(self, state: SpectrumState) -> jnp.ndarray:
        avg_channel_gain = jnp.mean(state.channel_gains, axis=0)[:, None]
        signal = state.tx_power - jnp.tile(avg_channel_gain, (1, self.num_bands))
        interference_db = 10 * jnp.log10(state.interference_map + 1e-9)
        return signal - interference_db

    def _calculate_reward(self, state: SpectrumState, previous_tx_power: jnp.ndarray) -> float:
        THROUGHPUT_COEFF, FAIRNESS_COEFF, POWER_COST_COEFF, SWITCHING_COST_COEFF, VIOLATION_PENALTY_COEFF = 5.0, 0.5, 0.1, 1.0, 10.0
        sinr_db = self._compute_sinr(state)
        sinr_linear = 10**(sinr_db / 10.0)
        is_active = state.tx_power > 0
        throughput_per_band = jnp.log2(1 + sinr_linear) * is_active
        total_throughput_reward = THROUGHPUT_COEFF * jnp.sum(throughput_per_band)
        sum_of_throughputs = jnp.sum(throughput_per_band)
        sum_of_squares = jnp.sum(throughput_per_band**2)
        num_active_bands = jnp.sum(is_active)
        fairness_denominator = num_active_bands * sum_of_squares + 1e-9
        fairness_reward = FAIRNESS_COEFF * (sum_of_throughputs**2) / fairness_denominator
        sinr_violations = jnp.sum(jnp.where(is_active, sinr_db < self.min_sinr, 0.0))
        latency_violations = jnp.sum(state.user_latency > self.max_latency)
        violation_penalty = VIOLATION_PENALTY_COEFF * (latency_violations + sinr_violations)
        power_cost = POWER_COST_COEFF * jnp.sum(state.tx_power)
        switching_cost = SWITCHING_COST_COEFF * jnp.sum(jnp.abs(previous_tx_power - state.tx_power))
        return total_throughput_reward + fairness_reward - violation_penalty - power_cost - switching_cost

    def _compute_metrics(self, state: SpectrumState) -> Dict[str, chex.Array]:
        """Computes a dictionary of performance metrics from a given state."""
        sinr_db = self._compute_sinr(state)
        sinr_linear = 10**(sinr_db / 10.0)
        is_active = state.tx_power > 0

        throughput_per_band = jnp.log2(1 + sinr_linear) * is_active
        total_throughput = jnp.sum(throughput_per_band)

        sum_of_throughputs = jnp.sum(throughput_per_band)
        sum_of_squares = jnp.sum(throughput_per_band**2)
        num_active_bands = jnp.sum(is_active)
        fairness_denominator = num_active_bands * sum_of_squares + 1e-9
        # Use jnp.divide for safe division
        fairness_index = jnp.divide(sum_of_throughputs**2, fairness_denominator)

        sinr_violations = jnp.sum(jnp.where(is_active, sinr_db < self.min_sinr, 0.0))
        latency_violations = jnp.sum(state.user_latency > self.max_latency)

        return {
            "total_throughput": total_throughput,
            "fairness_index": fairness_index,
            "sinr_violations": sinr_violations,
            "latency_violations": latency_violations
        }

    def _step_dynamics(self, state: SpectrumState, action: jnp.ndarray) -> SpectrumState:
        key, subkey = jax.random.split(state.key)
        new_channel = state.channel_gains * self.fading_coherence + \
                      (1 - self.fading_coherence) * jax.random.normal(subkey, state.channel_gains.shape)
        new_tx_power = self.power_levels[action]
        new_latency = state.user_latency + 1.0
        return SpectrumState(
            channel_gains=new_channel,
            interference_map=state.interference_map,
            user_latency=new_latency,
            spectrum_alloc=action,
            tx_power=new_tx_power,
            time=state.time + 1,
            key=key
        )

    def step(self, state: SpectrumState, action: chex.Array) -> Tuple[SpectrumState, TimeStep]:
        action = action.reshape(self.num_bs, self.num_bands)
        action_mask = self._mask_unsafe_actions(state)
        safe_action = jnp.where(action_mask, action, 0)
        previous_tx_power = state.tx_power
        new_state = self._step_dynamics(state, safe_action)
        reward = self._calculate_reward(new_state, previous_tx_power)
        terminated = jnp.any(new_state.user_latency > 2 * self.max_latency)
        truncated = new_state.time >= self.max_steps
        return new_state, jax.lax.cond(
            terminated,
            lambda s, r: termination(reward=r, observation=s),
            lambda s, r: jax.lax.cond(
                truncated,
                lambda s, r: truncation(reward=r, observation=s),
                lambda s, r: transition(reward=r, observation=s),
                s, r
            ),
            new_state, reward
        )

    def render(self, state: SpectrumState) -> None:
        sinr_db = self._compute_sinr(state)
        print(f"Step {int(state.time)}")
        print("Spectrum Allocation:")
        print(state.spectrum_alloc)
        print("Transmit Power (dBm):")
        print(state.tx_power)
        print(f"Latency violations: {jnp.sum(state.qos_metrics[:, 0] > self.max_latency)}")
        print(f"SINR violations: {jnp.sum(jnp.max(sinr_db, axis=1) < self.min_sinr)}")
        print(f"Average user SINR: {jnp.mean(jnp.max(sinr_db, axis=1)):.2f} dB")
        print(f"Average throughput: {jnp.mean(state.qos_metrics[:, 1]):.3f}")
        
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
    
    # Test SINR calculation
    sinr_values = env._compute_sinr(new_state)
    print(f"SINR shape: {sinr_values.shape}")
    print(f"Sample SINR values: {sinr_values[:3, :3]}")
    
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
