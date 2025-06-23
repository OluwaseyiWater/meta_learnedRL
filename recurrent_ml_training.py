from mLN.environment import DynamicSpectrumEnv
from models.recurrent_ml import make_networks, train_recurrent_maml_ppo
from utils import save_model, flatten_state
import jax
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pickle

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    key = jax.random.PRNGKey(cfg.seed if hasattr(cfg, 'seed') else 42) 
    key, init_key, train_key = jax.random.split(key, 3) 

    env = hydra.utils.instantiate(cfg.env)
    num_bs = env.num_bs
    num_bands = env.num_bands
    num_power_levels = env.num_power_levels

    state, _ = env.reset(key)
    sample_obs = flatten_state(state)
    obs_dim = sample_obs.shape[0]


    policy, value = make_networks(num_bs, num_bands, num_power_levels)
    policy_params = policy.init(init_key, sample_obs, None)
    policy_params = policy.init(init_key, sample_obs, None)
    value_params = value.init(init_key, sample_obs)

    conf_train = {
        "env": env,
        "policy_params": policy_params,
        "value_params": value_params,
        "policy_apply": policy.apply,
        "value_apply": value.apply,
        "num_tasks": cfg.recurrent_ml.num_tasks_per_batch,
        "inner_lr": cfg.recurrent_ml.inner_lr,
        "inner_steps": cfg.recurrent_ml.inner_steps,
        "meta_lr": cfg.recurrent_ml.meta_lr,
        "num_iterations": cfg.recurrent_ml.num_meta_iterations,
        "obs_dim": obs_dim,
        "key": key,
        "clip_ratio": cfg.recurrent_ml.clip_ratio,
        "wandb_project": cfg.recurrent_ml.wandb_project,
        "wandb_name": cfg.recurrent_ml.wandb_name,
        "use_wandb": cfg.recurrent_ml.use_wandb
    }

    trained_params, history = train_recurrent_maml_ppo(conf_train)

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    model_dir = os.path.join(output_dir, "models_output") 
    os.makedirs(model_dir, exist_ok=True)

    save_model(os.path.join(model_dir, "ppo_trained_params.pkl"), trained_params, None)
    with open(os.path.join(model_dir, "ppo_history.pkl"), 'wb') as f:
        pickle.dump(history, f)

    print(f"Trained parameters saved to {model_dir}/ppo_trained_params.pkl")
    print(f"Training history saved to {model_dir}/ppo_history.pkl")
    print(f"Number of base stations: {env.num_bs}")
    print(f"Number of bands: {env.num_bands}")
    print(f"Number of power levels: {env.num_power_levels}")
    print("Training completed.")

if __name__ == "__main__":
    main()

