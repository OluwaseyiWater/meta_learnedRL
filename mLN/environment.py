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
# Additional constants for proper wireless calculations
NOISE_FIGURE_DB = 7.0
BANDWIDTH_HZ = 10e6  # 10 MHz per band
THERMAL_NOISE_DBM_HZ = -174.0  # Thermal noise density

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
            qos_metrics=specs.Array((self.num_users, 2), jnp.float32),
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
            qos_metrics=jnp.zeros((self.num_users, 2), dtype=jnp.float32),
            spectrum_alloc=jnp.zeros((self.num_bs, self.num_bands), dtype=jnp.int32),
            tx_power=jnp.zeros((self.num_bs, self.num_bands), dtype=jnp.float32),
            time=jnp.array(0, dtype=jnp.int32),
            key=key
        )
        return state, restart(state)

    def _mask_unsafe_actions(self, state: SpectrumState) -> chex.Array:
        # Mask based on interference levels 
        interference_linear = 10 ** (state.interference_map / 10.0)  
        max_interference_linear = 10 ** (jnp.log10(self.max_interference * 1000) / 10.0)  
        interference_mask = interference_linear < max_interference_linear
        interference_mask = jnp.expand_dims(interference_mask, axis=-1)
        interference_mask = jnp.tile(interference_mask, (1, 1, self.num_power_levels))
        return interference_mask

    def _compute_sinr(self, state: SpectrumState) -> jnp.ndarray:
        # SINR calculation for wireless systems
        
        #  thermal noise power in dBm
        noise_power_dbm = (self.thermal_noise_dbm_hz + self.noise_figure_db + 
                          10 * jnp.log10(self.bandwidth_hz))
        
       
        sinr_db = jnp.zeros((self.num_users, self.num_bands))
        
        for user_idx in range(self.num_users):
            for band_idx in range(self.num_bands):
                received_powers = state.tx_power[:, band_idx] + state.channel_gains[user_idx, :]
                
                serving_bs = jnp.argmax(received_powers)
                signal_power_dbm = received_powers[serving_bs]
                # Interference from other BSs
                other_bs_mask = jnp.arange(self.num_bs) != serving_bs
                interference_from_bs = jnp.where(
                    other_bs_mask,
                    10 ** (received_powers / 10.0), 
                    0.0
                )
                total_interference_from_bs = jnp.sum(interference_from_bs)
                
                
                external_interference_linear = 10 ** (state.interference_map[serving_bs, band_idx] / 10.0)
                
                # Thermal noise (convert to linear)
                noise_power_linear = 10 ** (noise_power_dbm / 10.0)
                
                # Total interference + noise
                total_interference_linear = (total_interference_from_bs + 
                                           external_interference_linear + 
                                           noise_power_linear)
                
                #  signal power to linear
                signal_power_linear = 10 ** (signal_power_dbm / 10.0)
                
                # Calculate SINR and convert back to dB
                sinr_linear = signal_power_linear / (total_interference_linear + 1e-12)
                sinr_db = sinr_db.at[user_idx, band_idx].set(10 * jnp.log10(sinr_linear + 1e-12))
        
        return sinr_db

    def _calculate_reward(self, state: SpectrumState, action: jnp.ndarray, previous_tx_power: jnp.ndarray) -> float:
        POWER_COST_COEFF = 0.05  
        SWITCHING_COST_COEFF = 0.5  
        UTILIZATION_BONUS_COEFF = 1.0  
        VIOLATION_PENALTY_COEFF = 15.0  
        FAIRNESS_COEFF = 1.0  
        THROUGHPUT_COEFF = 2.0  #
        
        # Calculate SINR for all users
        sinr_db = self._compute_sinr(state)
        
        # Calculate throughput using Shannon capacity 
        # C = B * log2(1 + SINR_linear)
        sinr_linear = 10 ** (sinr_db / 10.0)
        best_sinr_per_user = jnp.max(sinr_linear, axis=1)
        
        # Spectral efficiency (bits/s/Hz)
        spectral_efficiency = jnp.log2(1 + best_sinr_per_user + 1e-12)
        # Total throughput (normalized)
        throughput = jnp.sum(spectral_efficiency) / self.num_users
        
        # Fairness using Jain's fairness index
        # F = (sum(xi))^2 / (n * sum(xi^2))
        throughput_per_user = spectral_efficiency + 1e-12  
        fairness = (jnp.sum(throughput_per_user)**2) / (self.num_users * jnp.sum(throughput_per_user**2))
        
        # SINR violations (users below minimum SINR threshold)
        sinr_violations = jnp.sum(jnp.max(sinr_db, axis=1) < self.min_sinr)
        
        # QoS violations (latency)
        latency_violations = jnp.sum(state.qos_metrics[:, 0] > self.max_latency)
        
        # Total violations penalty
        violation_penalty = VIOLATION_PENALTY_COEFF * (latency_violations + sinr_violations)
        
        # Power consumption cost (encourage efficiency)
        new_tx_power = self.power_levels[action]
        total_power_linear = jnp.sum(10 ** (new_tx_power / 10.0))  # Convert to linear and sum
        power_cost = POWER_COST_COEFF * jnp.log10(total_power_linear + 1e-12)  # Log scale
        
        # Switching cost (penalize frequent changes)
        power_changes = jnp.abs(previous_tx_power - new_tx_power)
        switching_cost = SWITCHING_COST_COEFF * jnp.sum(power_changes) / (self.num_bs * self.num_bands)
        
        # Utilization bonus (encourage using spectrum)
        active_channels = jnp.sum(action > 0)
        utilization_bonus = UTILIZATION_BONUS_COEFF * (active_channels / (self.num_bs * self.num_bands))
        
        # Combined reward
        total_reward = (THROUGHPUT_COEFF * throughput + 
                       FAIRNESS_COEFF * fairness + 
                       utilization_bonus - 
                       violation_penalty - 
                       power_cost - 
                       switching_cost)
        
        return total_reward

    def _adaptive_penalty(self, state: SpectrumState) -> jnp.ndarray:
        # Enhanced adaptive penalty considering both interference and SINR
        
        # High interference penalty
        interference_linear = 10 ** (state.interference_map / 10.0)
        high_interference_penalty = 0.1 * jnp.sum(
            interference_linear > (0.8 * 10 ** (jnp.log10(self.max_interference * 1000) / 10.0))
        )
        
        # Poor coverage penalty (users with very low SINR)
        sinr_db = self._compute_sinr(state)
        poor_coverage_penalty = 0.05 * jnp.sum(jnp.max(sinr_db, axis=1) < (self.min_sinr - 5.0))
        
        penalty = high_interference_penalty + poor_coverage_penalty
        return penalty

    def _step_dynamics(self, state: SpectrumState, action: jnp.ndarray) -> SpectrumState:
        key, subkey1, subkey2 = jax.random.split(state.key, 3)
        
        # Correlated fading: h[t+1] = ρ*h[t] + sqrt(1-ρ²)*w[t]
        fading_noise = jax.random.normal(subkey1, state.channel_gains.shape) * 2.0  # 2 dB std for fast fading
        correlation_factor = jnp.sqrt(1 - self.fading_coherence**2)
        new_channel = (self.fading_coherence * state.channel_gains + 
                      correlation_factor * fading_noise)
        
        # Update spectrum allocation and power
        new_alloc = action
        new_tx_power = self.power_levels[action]
        
        # Update QoS metrics more realistically
        # Latency increases based on poor SINR conditions
        sinr_db = self._compute_sinr(state)
        user_best_sinr = jnp.max(sinr_db, axis=1)
        
        # Latency penalty for poor SINR (exponential relationship)
        latency_penalty = jnp.where(
            user_best_sinr < self.min_sinr,
            2.0 * jnp.exp(-(user_best_sinr - self.min_sinr) / 5.0),  # Exponential penalty
            0.1  # Small base latency
        )
        
        new_latency = state.qos_metrics[:, 0] + latency_penalty
        
        # Throughput based on actual spectral efficiency
        sinr_linear = 10 ** (user_best_sinr / 10.0)
        spectral_efficiency = jnp.log2(1 + sinr_linear + 1e-12)
        # Normalize throughput to reasonable range
        normalized_throughput = spectral_efficiency / 10.0  # Normalize by ~10 bits/s/Hz max
        
        # Update QoS metrics
        new_qos = jnp.stack([new_latency, normalized_throughput], axis=1)
        
        # Evolve external interference slightly 
        interference_noise = jax.random.normal(subkey2, state.interference_map.shape) * 0.5  
        new_interference_map = state.interference_map + interference_noise
       
        new_interference_map = jnp.clip(new_interference_map, -20.0, 30.0)  
        
        return SpectrumState(
            channel_gains=new_channel,
            interference_map=new_interference_map,
            qos_metrics=new_qos,
            spectrum_alloc=new_alloc,
            tx_power=new_tx_power,
            time=state.time + 1,
            key=key
        )

    def step(self, state: SpectrumState, action: chex.Array) -> Tuple[SpectrumState, TimeStep]:
        action = action.reshape(self.num_bs, self.num_bands)
        action_mask = self._mask_unsafe_actions(state)
        
        #  safety mask
        safety = action_mask[jnp.arange(self.num_bs)[:, None], 
                            jnp.arange(self.num_bands)[None, :], 
                            action]
        safe_action = jnp.where(safety, action, 0)  
        
        previous_tx_power = state.tx_power
        new_state = self._step_dynamics(state, safe_action)
        reward = self._calculate_reward(new_state, safe_action, previous_tx_power) - self._adaptive_penalty(new_state)
        
        # Termination conditions
        terminated = jnp.any(new_state.qos_metrics[:, 0] > 2 * self.max_latency)
        sinr_db = self._compute_sinr(new_state)
        severe_sinr_violations = jnp.sum(jnp.max(sinr_db, axis=1) < (self.min_sinr - 10.0)) > (0.8 * self.num_users)
        terminated = jnp.logical_or(terminated, severe_sinr_violations)
        
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
