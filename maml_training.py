from mLN.environment import DynamicSpectrumEnv 
from models.maml import train_maml 

from utils import save_model
import jax
from models.maml import make_networks 

from utils import flatten_state
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pickle

# Hydra decorator to load config from a YAML file
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))

    env = hydra.utils.instantiate(cfg.env)
    num_bs = env.num_bs
    num_bands = env.num_bands
    num_power_levels = env.num_power_levels

    key = jax.random.PRNGKey(cfg.seed if hasattr(cfg, 'seed') else 42) 
    key, init_key, train_key = jax.random.split(key, 3) 

    # Initialize networks
    state, _ = env.reset(init_key) 
    sample_obs = flatten_state(state)
    policy, value = make_networks(num_bs, num_bands, num_power_levels)
    policy_params = policy.init(init_key, sample_obs) 
    value_params = value.init(init_key, sample_obs)   

    obs_dim = sample_obs.shape[0]


    trained_params, history = train_maml(
        env=env, 
        policy_params=policy_params,
        value_params=value_params,
        policy_apply=policy.apply,
        value_apply=value.apply,
        num_tasks=cfg.maml.num_meta_tasks, 
        inner_steps=cfg.maml.inner_steps,
        meta_lr=cfg.maml.meta_lr,
        inner_lr=cfg.maml.inner_lr,
        num_iterations=cfg.maml.num_meta_iterations,
        dim=obs_dim, # Observation dimension
        key=train_key, 
        eval_interval=cfg.maml.get('eval_interval', 10), 
        num_eval_tasks=cfg.maml.get('num_eval_tasks', 5),  
        wandb_project=cfg.maml.get('wandb_project', "maml-training"),
        wandb_name=cfg.maml.get('wandb_name', None), 
        use_wandb=cfg.maml.get('use_wandb', True)
    )

    # Get the output directory managed by Hydra
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    model_dir = os.path.join(output_dir, "models") 
    os.makedirs(model_dir, exist_ok=True)

    # Save the trained parameters
    save_path = os.path.join(model_dir, "trained_params.pkl")
    save_model(save_path, trained_params, None) 
    print(f"Trained parameters saved to {save_path}")

    # Save the training history
    history_path = os.path.join(model_dir, "history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Training history saved to {history_path}")

    # Print environment details
    print(f"Number of base stations: {env.num_bs}")
    print(f"Number of bands: {env.num_bands}")
    print(f"Number of power levels: {env.num_power_levels}")
    print("Training completed.")


if __name__ == "__main__":
    main()
