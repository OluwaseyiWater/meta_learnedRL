from mLN.environment import DynamicSpectrumEnv
from models.maml import train_maml, make_networks
from utils import save_model, flatten_state
import jax
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pickle

# Hydra decorator to load config from a YAML file
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Use Hydra's instantiate to create the environment
    env = hydra.utils.instantiate(cfg.env)
    
    # Ensure compatibility by setting additional attributes if needed
    if not hasattr(env, 'bandwidth_hz'):
        env.bandwidth_hz = getattr(env, 'bandwidth_hz', 10e6)
    if not hasattr(env, 'noise_figure_db'):
        env.noise_figure_db = getattr(env, 'noise_figure_db', 7.0)
    if not hasattr(env, 'thermal_noise_dbm_hz'):
        env.thermal_noise_dbm_hz = getattr(env, 'thermal_noise_dbm_hz', -174.0)
    
    # For compatibility with different attribute names in the code
    if hasattr(env, 'max_power_dbm'):
        env.max_power = env.max_power_dbm
    elif hasattr(env, 'max_power'):
        env.max_power_dbm = env.max_power
        
    if hasattr(env, 'power_levels_dbm'):
        env.power_levels = env.power_levels_dbm
    elif hasattr(env, 'power_levels'):
        env.power_levels_dbm = env.power_levels
        
    if hasattr(env, 'max_external_interference_mW'):
        env.max_interference = env.max_external_interference_mW
    elif hasattr(env, 'max_interference'):
        env.max_external_interference_mW = env.max_interference
        
    if hasattr(env, 'min_sinr_db'):
        env.min_sinr = env.min_sinr_db
    elif hasattr(env, 'min_sinr'):
        env.min_sinr_db = env.min_sinr
    
    num_bs = env.num_bs
    num_bands = env.num_bands
    num_power_levels = env.num_power_levels

    key = jax.random.PRNGKey(cfg.seed)
    key, init_key = jax.random.split(key)
    state, _ = env.reset(key)
    sample_obs = flatten_state(state)

    policy, value = make_networks(num_bs, num_bands, num_power_levels)
    policy_params = policy.init(init_key, sample_obs)
    value_params = value.init(init_key, sample_obs)
        
    trained_params, history = train_maml({
        "env":env,
        "policy_params":policy_params,
        "value_params":value_params,
        "policy_apply":policy.apply,
        "value_apply":value.apply,
        "num_tasks":cfg.maml.num_meta_tasks,
        "inner_steps":cfg.maml.inner_steps,
        "meta_lr":cfg.maml.meta_lr,
        "inner_lr":cfg.maml.inner_lr,
        "num_iterations":cfg.maml.num_meta_iterations,
        "dim":sample_obs.shape[0],
        "key":key,
    })

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    model_dir = os.path.join(output_dir, "models") 
    os.makedirs(model_dir, exist_ok=True)

    save_model(os.path.join(model_dir, "A2C_trained_params.pkl"), trained_params, None)
    with open(os.path.join(model_dir, "A2C_history.pkl"), 'wb') as f:
        pickle.dump(history, f)

    print(f"Trained parameters saved to {model_dir}/A2C_trained_params.pkl")
    print(f"Training history saved to {model_dir}/A2C_history.pkl")
    print(f"Number of base stations: {env.num_bs}")
    print(f"Number of bands: {env.num_bands}")
    print(f"Number of power levels: {env.num_power_levels}")
    print("Training completed.")


if __name__ == "__main__":
    main()



# Hydra decorator to load config from a YAML file
# @hydra.main(config_path="conf", config_name="config", version_base=None)
# def main(cfg: DictConfig) -> None:
    
#     print(OmegaConf.to_yaml(cfg))

#     env = hydra.utils.instantiate(cfg.env)
#     num_bs = env.num_bs
#     num_bands = env.num_bands
#     num_power_levels = env.num_power_levels
#     key = jax.random.PRNGKey(42)
#     key, init_key = jax.random.split(key)
#     state, _ = env.reset(key)
#     sample_obs = flatten_state(state)
#     policy, value = make_networks(num_bs, num_bands, num_power_levels)
#     policy_params = policy.init(init_key, sample_obs)
#     value_params = value.init(init_key, sample_obs)

#     obs_dim = sample_obs.shape[0]
    

#     trained_params, history = train_maml(
#         env,
#         policy_params=policy_params,
#         value_params=value_params,
#         policy_apply=policy.apply,
#         value_apply=value.apply,
#         num_tasks=cfg.maml.num_meta_tasks,
#         inner_steps=cfg.maml.inner_steps,
#         meta_lr=cfg.maml.meta_lr,
#         inner_lr=cfg.maml.inner_lr,
#         num_iterations=cfg.maml.num_meta_iterations,
#         dim=obs_dim,
#         key=key,
#     )

#     # Get the output directory managed by Hydra
#     output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
#     model_dir = os.path.join(output_dir, "models")
#     os.makedirs(model_dir, exist_ok=True)

#     # Save the trained parameters
#     save_path = os.path.join(model_dir, "trained_params.pkl")
#     save_model(save_path, trained_params, None)
#     print(f"Trained parameters saved to {save_path}")
    
#     # Save the training history
#     history_path = os.path.join(model_dir, "history.pkl")
#     with open(history_path, 'wb') as f:
#         pickle.dump(history, f)
#     print(f"Training history saved to {history_path}")

#     # Print environment details
#     print(f"Number of base stations: {env.num_bs}")
#     print(f"Number of bands: {env.num_bands}")
#     print(f"Number of power levels: {env.num_power_levels}")
#     print("Training completed.")


# if __name__ == "__main__":
#     main()