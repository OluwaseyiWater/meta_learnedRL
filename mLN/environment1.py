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
MAX_INTERFERENCE = 25.0  # dBm
MAX_POWER = 23.0  # dBm (maximum allowed transmit power)
MIN_SINR = 5.0  # dB (minimum required SINR)
MAX_LATENCY = 50.0  # ms (maximum allowed latency)
NUM_BS = 3
NUM_USERS = 10
NUM_BANDS = 5
NUM_POWER_LEVELS = 4  # Number of discrete power levels
POWER_LEVELS = jnp.linspace(0, MAX_POWER, NUM_POWER_LEVELS)  # Discrete power levels
FADING_COHERENCE = 0.9  # Time-correlation in fading
MAX_STEPS = 100

class DynamicSpectrumEnv(Environment):
    def __init__(self, num_bs=NUM_BS, num_users=NUM_USERS, num_bands=NUM_BANDS, max_steps=MAX_STEPS \
                 , max_latency=MAX_LATENCY, max_power=MAX_POWER, num_power_levels=NUM_POWER_LEVELS, power_levels=POWER_LEVELS \
                 , fading_coherence=FADING_COHERENCE, max_interference=MAX_INTERFERENCE, min_sinr=MIN_SINR):
        """
        Initialize the environment.

        Args:
            num_bs (int): Number of base stations.
            num_users (int): Number of users.
            num_bands (int): Number of frequency bands.
            max_steps (int): Maximum number of steps per episode.
            max_latency (float): Maximum allowed latency in milliseconds.
            max_power (float): Maximum allowed transmit power in dBm.
            num_power_levels (int): Number of discrete power levels.
            power_levels (jnp.ndarray): Array of discrete power levels in dBm.
            fading_coherence (float): Time-correlation in fading.
            max_interference (float): Maximum allowed interference in dBm.
            min_sinr (float): Minimum required SINR in dB.

        Returns:
            None
        """
        super().__init__()
        self._rng = jr.PRNGKey(0)
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
        # Action: [BS index, Band index, Power level index]
        return specs.MultiDiscreteArray(
            num_values=jnp.array([self.num_bs, self.num_bands, self.num_power_levels], dtype=jnp.int32),
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
        # Split keys for independent random operations
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        # Choose a scenario: 0 = Urban, 1 = Rural, 2 = Jamming scenario.
        scenario = jax.random.randint(subkey1, (), 0, 3)

        # Define path loss for different scenarios (simplified)
        path_loss = jnp.select(
            [scenario == 0, scenario == 1, scenario == 2],
            [128.1 + 37.6 * jnp.log10(0.5), 98.5 + 23.1 * jnp.log10(2.0), 105.3 + 34.2 * jnp.log10(1.0)]
        )

        # Initialize channel gains: shape (num_users, num_bs)
        channel_gains = path_loss + jax.random.normal(subkey2, (self.num_users, self.num_bs))
        # For interference, we now define it per BS and band.
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
        """
        Create an action mask of shape (num_bs, num_bands, NUM_POWER_LEVELS).
        For each candidate action, check if increasing the power to the chosen level
        would exceed MAX_POWER and if the current interference is below threshold.
        """
        # Expand current tx_power to candidate power levels:
        # For each BS and band, candidate new power = POWER_LEVELS (broadcasted)
        current_power = state.tx_power  # shape (num_bs, num_bands)
        # Create candidate power array of shape (num_bs, num_bands, NUM_POWER_LEVELS)
        candidate_power = jnp.broadcast_to(current_power[:, :, None], (self.num_bs, self.num_bands, self.num_power_levels))
        # Determine if candidate power (here we simply use the discrete levels) is allowed
        power_mask = candidate_power + self.power_levels[None, None, :] <= self.max_power

        # Check interference: assume interference_map is per (bs, band)
        # We allow action only if the interference is below MAX_INTERFERENCE threshold.
        interference_mask = state.interference_map[:, :, None] < MAX_INTERFERENCE

        # Combine masks elementwise
        action_mask = jnp.logical_and(power_mask, interference_mask)
        return action_mask

    def _compute_sinr(self, state: SpectrumState) -> jnp.ndarray:
        """
        Compute a simplified SINR per BS and band.
        Here, we assume that the effective signal is given by the tx_power minus channel loss (averaged over users),
        and interference is taken from the interference map.
        """
        # Average channel gain from BS to users: shape (num_bs,)
        avg_channel_gain = jnp.mean(state.channel_gains, axis=0)  # shape: (num_bs,)
        # Expand dimensions to match tx_power shape (num_bs, num_bands)
        avg_channel_gain = jnp.tile(avg_channel_gain[:, None], (1, self.num_bands))
        signal = state.tx_power - avg_channel_gain  # dB difference (simplified)
        # Compute interference in dB scale (using a small epsilon to avoid log of zero)
        interference_db = 10 * jnp.log10(state.interference_map + 1e-6)
        # Simplified SINR: signal minus interference
        sinr = signal - interference_db
        return sinr  # shape: (num_bs, num_bands)

    # def _calculate_reward(self, state: SpectrumState) -> jnp.ndarray:
    #     """
    #     Calculate reward based on throughput, fairness and penalties for QoS violations.
    #     This is a simplified placeholder function.
    #     """
    #     # Throughput: sum over all users (from qos_metrics column 1)
    #     throughput = jnp.sum(state.qos_metrics[:, 1])
    #     # Fairness: inversely related to the variance of throughput
    #     fairness = 1.0 / (1e-6 + jnp.var(state.qos_metrics[:, 1]))
    #     # SINR calculation and count violations (simplified)
    #     sinr = self._compute_sinr(state)
    #     sinr_violations = jnp.sum(sinr < MIN_SINR)
    #     # Latency violations: count users whose latency exceeds MAX_LATENCY
    #     latency_violations = jnp.sum(state.qos_metrics[:, 0] > MAX_LATENCY)
    #     reward = throughput + fairness - 10.0 * (latency_violations + sinr_violations)
    #     return reward

    def _calculate_reward(self, state: SpectrumState, action: chex.Array) -> float:
        """
        Compute a composite reward that includes both state-dependent and action-dependent components.
    
        State-dependent components:
          - Throughput: Encourages high total throughput across all users.
          - Fairness: Rewards low variance in user throughput.
          - Violation penalties: Penalizes states where latency exceeds MAX_LATENCY or SINR falls below MIN_SINR.
          
        Action-dependent components:
          - Power cost: Penalizes selecting high power levels.
          - Switching cost: Penalizes abrupt changes in power (to discourage frequent switching).
          - Utilization bonus: Rewards actions that maintain or enable spectrum utilization.
    
        Parameters:
          state: The current SpectrumState, which contains the network metrics.
          action: A discrete action array [bs_index, band_index, power_level_index].
    
        Returns:
          A scalar reward (float) that balances these components.
        """
        # Coefficients for tuning the reward signal
        POWER_COST_COEFF = 0.1       # Cost per unit of power level (action-dependent)
        SWITCHING_COST_COEFF = 1.0   # Cost for switching power levels (action-dependent)
        UTILIZATION_BONUS_COEFF = 2.0  # Bonus for utilizing allocated spectrum (action-dependent)
        VIOLATION_PENALTY_COEFF = 10.0 # Penalty per violation (state-dependent)
    
        # --- Action Decomposition ---
        bs, band, power_level = action[0], action[1], action[2]
    
        # --- State-Dependent Rewards ---
        # Throughput: Sum of throughput for all users (assumed to be at index 1 in qos_metrics)
        throughput = jnp.sum(state.qos_metrics[:, 1])
        
        # Fairness: Inverse of variance in throughput among users (higher fairness if variance is lower)
        fairness = 1.0 / (1e-6 + jnp.var(state.qos_metrics[:, 1]))
        
        # Compute SINR from the state; violations occur when SINR is below the threshold
        sinr = self._compute_sinr(state)
        sinr_violations = jnp.sum(sinr < self.min_sinr)
        
        # Latency violations: Count users with latency above the allowed maximum
        latency_violations = jnp.sum(state.qos_metrics[:, 0] > self.max_latency)
        
        # Total penalty for state violations
        violation_penalty = VIOLATION_PENALTY_COEFF * (latency_violations + sinr_violations)
    
        # --- Action-Dependent Costs and Bonuses ---
        prev_power = state.tx_power[bs, band]
        
        # Power cost: Penalize high power level selections
        power_cost = POWER_COST_COEFF * power_level
    
        # Switching cost: Penalize changes in power level (to discourage rapid switching)
        switching_cost = SWITCHING_COST_COEFF * jnp.abs(prev_power - power_level)
        
        # Utilization bonus: Reward actions that maintain spectrum allocation.
        utilization_bonus = UTILIZATION_BONUS_COEFF * jnp.where(state.spectrum_alloc[bs, band] > 0, 1.0, 0.0)
        
        # --- Total Reward Calculation ---
        total_reward = throughput + fairness + utilization_bonus - violation_penalty - power_cost - switching_cost
        
        return total_reward


    def _adaptive_penalty(self, state: SpectrumState) -> jnp.ndarray:
        # Placeholder: apply a small penalty for high interference values.
        penalty = 0.1 * jnp.sum(state.interference_map > (0.8 * MAX_INTERFERENCE))
        return penalty

    def _step_dynamics(self, state: SpectrumState, action: jnp.ndarray) -> SpectrumState:
        """
        Update the state based on the (safe) action.
        Action is assumed to be [bs_index, band_index, power_level_index].
        """
        # Split key for new randomness
        key, subkey = jax.random.split(state.key)
        # Update channel gains using a block-fading model (simplified)
        new_channel = state.channel_gains * self.fading_coherence + jax.random.normal(subkey, state.channel_gains.shape)
        
        # Update spectrum allocation and tx_power for the chosen BS and band.
        bs_idx, band_idx, power_idx = action[0], action[1], action[2]
        new_alloc = state.spectrum_alloc.at[bs_idx, band_idx].set(power_idx)
        # Update transmit power based on the chosen discrete power level.
        new_tx_power = state.tx_power.at[bs_idx, band_idx].set(POWER_LEVELS[power_idx])
        
        # Simplified QoS update:
        # Increase latency for all users by one time unit.
        new_qos = state.qos_metrics.at[:, 0].add(1.0)
        # For throughput, add a bonus to users served by the selected BS (simplified allocation)
        # Here we assume that users are uniformly associated with BS indices.
        users_served = jnp.arange(self.num_users)  # Placeholder for association logic
        throughput_bonus = 1.0 * (self.power_levels[power_idx] / self.max_power)
        new_throughput = new_qos[:, 1] + throughput_bonus  # Add bonus uniformly
        new_qos = new_qos.at[:, 1].set(new_throughput)
        
        return SpectrumState(
            channel_gains=new_channel,
            interference_map=state.interference_map,  # Keeping interference static in this simplified model
            qos_metrics=new_qos,
            spectrum_alloc=new_alloc,
            tx_power=new_tx_power,
            time=state.time + 1,
            key=key
        )

    def step(self, state: SpectrumState, action: chex.Array) -> Tuple[SpectrumState, TimeStep]:
        action_mask = self._mask_unsafe_actions(state)
        safe_action = jnp.where(action_mask[action[0], action[1], action[2]], 
                              action, jnp.array([0, 0, 0]))
        
        new_state = self._step_dynamics(state, safe_action)
        reward = self._calculate_reward(new_state, safe_action) - self._adaptive_penalty(new_state)

        # Evaluate termination and truncation conditions and convert to Python booleans.
        terminated = bool(jnp.any(new_state.qos_metrics[:, 0] > 2 * MAX_LATENCY))
        truncated  = bool(new_state.time >= self.max_steps)
        
        # Combined done flag.
        done_flag = terminated or truncated
    
        # Use jax.lax.cond with the scalar boolean
        timestep = jax.lax.cond(
            done_flag,
            lambda: jax.lax.cond(
                bool(terminated),
                lambda: termination(reward, new_state),  # discount=0.0
                lambda: truncation(reward, new_state),    # discount=1.0
            ),
            lambda: transition(reward, new_state),        # discount=1.0
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



@chex.dataclass
class SpectrumState:
    channel_gains: chex.Array      # Shape: (num_bs, NUM_USERS) or (NUM_USERS, num_bs)
    interference_map: chex.Array   # Now per BS and band: shape (num_bs, num_bands)
    qos_metrics: chex.Array        # Shape: (NUM_USERS, 2) [latency, throughput]
    spectrum_alloc: chex.Array     # Shape: (num_bs, num_bands), discrete allocation indices
    tx_power: chex.Array           # Shape: (num_bs, num_bands), current transmit power (dBm)
    time: chex.Array               # Scalar step counter
    key: chex.PRNGKey              # JAX PRNG key for randomness

