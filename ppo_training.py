import jax
import jax.numpy as jnp
import optax
import haiku as hk
from collections import namedtuple
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pickle
import wandb 

from typing import Any, Optional, Dict, Tuple 

from hydra.core.hydra_config import HydraConfig 


from models.ppo import (
    DynamicSpectrumEnv, 
    make_recurrent_networks_ppo,
    init_obs_normalizer,
    update_obs_normalizer,
    sample_trajectories_ppo,
    compute_gae_ppo,
    ppo_update_step,
    flatten_state, 
    SpectrumState 
)
def save_model(path: str, params: Any, opt_state: Optional[Any] = None): 
    """Saves model parameters and optionally optimizer state."""
    data_to_save = {"params": params}
    if opt_state is not None:
        data_to_save["opt_state"] = opt_state
    with open(path, "wb") as f:
        pickle.dump(data_to_save, f)
    print(f"Model saved to {path}")


PPOConfig = namedtuple("PPOConfig", [
    "seed", "learning_rate_actor", "learning_rate_critic", "num_epochs", "num_minibatches",
    "rollout_length_ppo", "vf_coef", "ent_coef", "clip_epsilon", "max_grad_norm",
    "gamma", "lambda_", "use_wandb", "wandb_project", "wandb_name_ppo", 
    "log_interval_ppo", "total_timesteps", 
    "lstm_hidden_dim", "mlp_hidden_dim", "critic_hidden_dim",
    "save_model_name_ppo", "save_history_name_ppo" 
])


def train_ppo(config: PPOConfig):
    """Main PPO training function."""

    if config.use_wandb:
        try:
            wandb.init(project=config.wandb_project, name=config.wandb_name_ppo, config=dict(config._asdict()))
        except Exception as e:
            print(f"Warning: Failed to initialize WandB: {e}")

    env = DynamicSpectrumEnv() 
    
    # Network dimensions and action space details
    num_total_actions = env.num_bs * env.num_bands
    num_power_options = env.num_power_levels
    actor_output_size = num_total_actions * num_power_options

    key = jax.random.PRNGKey(config.seed)
    key_actor, key_critic, key_init_actor_opt, key_init_critic_opt, key_rollout = jax.random.split(key, 5)

    actor_net, critic_net = make_recurrent_networks_ppo(actor_output_size, config)

    # Initialize networks
    dummy_spec_state = env.observation_spec().generate_value()
    dummy_flat_obs = flatten_state(dummy_spec_state)
    dummy_actor_input_obs = dummy_flat_obs[None, :] # Add batch dim for LSTM input

    actor_params = actor_net.init(key_actor, dummy_actor_input_obs, None)
    critic_params = critic_net.init(key_critic, dummy_flat_obs) 

    # Initialize Optimizers
    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate_actor, eps=1e-5) 
    )
    critic_optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate_critic, eps=1e-5) 
    )
    actor_opt_state = actor_optimizer.init(actor_params)
    critic_opt_state = critic_optimizer.init(critic_params)

    # Initialize Observation Normalizer
    obs_norm_state = init_obs_normalizer(dummy_flat_obs.shape[0])

    # Training loop
    num_updates = config.total_timesteps // config.rollout_length_ppo
    all_metrics_history = []

    print(f"Starting PPO training for {config.total_timesteps} total timesteps.")
    print(f"Rollout length: {config.rollout_length_ppo}, Num updates: {num_updates}")

    for update_iter in range(1, num_updates + 1):
        key_rollout, current_iter_key = jax.random.split(key_rollout)

        #  Collect Trajectories
        trajectory_data = sample_trajectories_ppo(
            env, actor_params, critic_params, actor_net.apply, critic_net.apply,
            current_iter_key, obs_norm_state, config.rollout_length_ppo,
            num_total_actions, num_power_options
        )
        
        # Update Observation Normalizer 
        current_obs_for_norm_update = trajectory_data["obs"]
        obs_norm_state_after_sampling_and_before_update = obs_norm_state 
        obs_norm_state = update_obs_normalizer(obs_norm_state_after_sampling_and_before_update, current_obs_for_norm_update)
        
        #  Compute Advantages and Returns
        returns, advantages = compute_gae_ppo(trajectory_data, config.gamma, config.lambda_)
        trajectory_data["returns"] = returns
        trajectory_data["advantages"] = advantages
        
        #  PPO Update Epochs
        num_samples_in_trajectory = trajectory_data["obs"].shape[0]
        actual_num_minibatches = max(1, config.num_minibatches)
        batch_size = num_samples_in_trajectory // actual_num_minibatches
        
        if batch_size == 0 and num_samples_in_trajectory > 0 : 
            batch_size = num_samples_in_trajectory
            actual_num_minibatches = 1
            print(f"Warning: rollout_length_ppo ({config.rollout_length_ppo}) is less than num_minibatches ({config.num_minibatches}). Setting batch_size to rollout_length and num_minibatches to 1 for this update.")


        if "log_probs" not in trajectory_data:
            raise ValueError("log_probs missing from trajectory_data. Ensure sample_trajectories_ppo returns it.")

        accumulated_epoch_metrics = {}

        for epoch_num in range(config.num_epochs):
            key_rollout, perm_key = jax.random.split(key_rollout) 
            permutation = jax.random.permutation(perm_key, num_samples_in_trajectory)
            
            def is_scalar_leaf(x):
                if isinstance(x, jnp.ndarray) and x.ndim == 0: 
                    return True
                return False

    
            shuffled_trajectory = jax.tree.map(
                lambda x: x[permutation] if not is_scalar_leaf(x) else x,
                trajectory_data
            )

            for mb_start in range(0, num_samples_in_trajectory, batch_size):
                mb_end = mb_start + batch_size
                if mb_end > num_samples_in_trajectory : 
                    mb_end = num_samples_in_trajectory
                if mb_start == mb_end: 
                    continue

                minibatch_data = jax.tree.map(
                    lambda x: x[mb_start:mb_end] if not is_scalar_leaf(x) else x, 
                    shuffled_trajectory
                )
                
                if minibatch_data["obs"].shape[0] == 0:
                    continue

                actor_params, critic_params, actor_opt_state, critic_opt_state, update_metrics = \
                    ppo_update_step(
                        actor_params, critic_params, actor_opt_state, critic_opt_state,
                        minibatch_data,
                        obs_norm_state_after_sampling_and_before_update, 
                        actor_net.apply, critic_net.apply,
                        actor_optimizer, critic_optimizer,
                        config, num_total_actions, num_power_options
                    )
                
                for k, v in update_metrics.items():
                    accumulated_epoch_metrics[k] = accumulated_epoch_metrics.get(k, 0.0) + v

        num_total_minibatches_processed = actual_num_minibatches * config.num_epochs
        avg_epoch_metrics = {k: v / num_total_minibatches_processed
                             for k, v in accumulated_epoch_metrics.items()
                             if num_total_minibatches_processed > 0} 
        
        all_metrics_history.append(avg_epoch_metrics)

        if update_iter % config.log_interval_ppo == 0 and avg_epoch_metrics:
            print(f"[Update {update_iter}/{num_updates}] Timesteps: {update_iter * config.rollout_length_ppo}")
            log_str = ""
            for k, v in avg_epoch_metrics.items():
                log_str += f"{k}: {v:.4f} | "
            if log_str: print(log_str[:-3]) 
            
            if config.use_wandb:
                wandb.log({"update_iteration": update_iter, 
                           "timesteps_total": update_iter * config.rollout_length_ppo,
                           **avg_epoch_metrics})
    
    
    output_dir = HydraConfig.get().runtime.output_dir
    model_dir = os.path.join(output_dir, "models_output_ppo") 
    os.makedirs(model_dir, exist_ok=True)

    final_params_to_save = {"actor_params": actor_params, "critic_params": critic_params, "obs_norm_state": obs_norm_state}
    save_model(os.path.join(model_dir, config.save_model_name_ppo), final_params_to_save, None)

    with open(os.path.join(model_dir, config.save_history_name_ppo), "wb") as f:
        pickle.dump(all_metrics_history, f)

    print(f"Trained PPO parameters saved to {model_dir}/{config.save_model_name_ppo}")
    print(f"PPO training history saved to {model_dir}/{config.save_history_name_ppo}")

    if config.use_wandb:
        wandb.finish()
    
    return final_params_to_save, all_metrics_history


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig): 
    
    if "ppo" not in cfg:
        print("Error: 'ppo' configuration section not found in config.yaml.")
        print("Please add a 'ppo' section to your Hydra configuration.")
        return

    ppo_cfg_dict = OmegaConf.to_container(cfg.ppo, resolve=True)
    
    required_keys = set(PPOConfig._fields)
    actual_keys = set(ppo_cfg_dict.keys())
    
    if 'seed' not in actual_keys and 'seed' in cfg:
        ppo_cfg_dict['seed'] = cfg.seed

    missing_keys = required_keys - actual_keys
    if missing_keys:
        print(f"Error: Missing keys in PPO configuration: {missing_keys}")
        print(f"Please ensure your 'ppo' section in config.yaml has all fields defined in PPOConfig.")
        return

    config_for_ppo_training = PPOConfig(**ppo_cfg_dict)

    print("Starting PPO training with JAX...")
    print(OmegaConf.to_yaml(cfg.ppo)) 

    train_ppo(config_for_ppo_training)

    print("PPO Training completed.")

if __name__ == "__main__":
    main()
