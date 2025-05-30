import jax
from models.recurrent_ml import make_networks, train_recurrent_maml_ppo
from mLN.environment import DynamicSpectrumEnv 
from utils import flatten_state, save_model 
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

    # Initialize networks
    state, _ = env.reset(init_key)
    sample_obs = flatten_state(state)
    obs_dim = sample_obs.shape[0]

    policy, value = make_networks(num_bs, num_bands, num_power_levels)
    policy_params = policy.init(init_key, sample_obs, None)
    value_params = value.init(init_key, sample_obs)

    params, loss_history = train_recurrent_maml_ppo(
        env=env,
        policy_params=policy_params,
        value_params=value_params,
        policy_apply=policy.apply,
        value_apply=value.apply,
        num_tasks=cfg.recurrent_ml.num_tasks_per_batch,
        inner_lr=cfg.recurrent_ml.inner_lr,
        inner_steps=cfg.recurrent_ml.inner_steps,
        meta_lr=cfg.recurrent_ml.meta_lr,
        num_iterations=cfg.recurrent_ml.num_meta_iterations,
        obs_dim=obs_dim,
        key=train_key, 
        clip_ratio=cfg.recurrent_ml.clip_ratio,
        eval_interval=cfg.recurrent_ml.get('eval_interval', 10), 
        num_eval_tasks=cfg.recurrent_ml.get('num_eval_tasks', 5), 
        wandb_project=cfg.recurrent_ml.wandb_project,
        wandb_name=cfg.recurrent_ml.wandb_name,
        use_wandb=cfg.recurrent_ml.use_wandb,
        rollout_len=cfg.recurrent_ml.get('rollout_len', 50) 
    )

    # Save the trained parameters
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    model_dir = os.path.join(output_dir, "models_output") 
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "trained_params.pkl")
    save_model(save_path, params, None) 
    print(f"Trained parameters saved to {save_path}")

    # Save the training history
    history_path = os.path.join(model_dir, "history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(loss_history, f)
    print(f"Training history saved to {history_path}")

    # Print environment details
    print(f"Number of base stations: {env.num_bs}")
    print(f"Number of bands: {env.num_bands}")
    print(f"Number of power levels: {env.num_power_levels}")
    print("Training completed.")

if __name__ == "__main__":
    main()
