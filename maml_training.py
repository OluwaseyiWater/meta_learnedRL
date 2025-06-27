from mLN.environment import DynamicSpectrumEnv
from models.maml import train_maml
from utils import save_model
import jax
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import os
import pickle


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Starting MAML training...")
    print(OmegaConf.to_yaml(cfg.maml))

    config = {
        "seed": cfg.seed,
        "meta_lr": cfg.maml.meta_lr,
        "inner_policy_lr": cfg.maml.inner_policy_lr,
        "inner_value_lr": cfg.maml.inner_value_lr,
        "inner_steps": cfg.maml.inner_steps,
        "meta_batch_size": cfg.maml.meta_batch_size,
        "num_meta_iters": cfg.maml.num_meta_iters,
        "rollout_length": cfg.maml.rollout_length,
        "vf_coef": cfg.maml.vf_coef,
        "ent_coef": cfg.maml.ent_coef,
        "eval_interval": cfg.maml.eval_interval,
        "num_eval_tasks": cfg.maml.num_eval_tasks,
        "max_grad_norm": cfg.maml.max_grad_norm,
        "log_interval": cfg.maml.log_interval,
        "use_wandb": cfg.maml.use_wandb,
        "wandb_project": cfg.maml.wandb_project,
        "wandb_name": cfg.maml.wandb_name,
    }

    # Train and capture history
    trained_params, history = train_maml(config)

    # Save paths
    output_dir = HydraConfig.get().runtime.output_dir
    model_dir = os.path.join(output_dir, "models_output")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "maml_trained_params.pkl")
    history_path = os.path.join(model_dir, "maml_history.pkl")

    # Save model + history
    save_model(model_path, trained_params, None)
    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    print(f"✅ Trained model saved to: {model_path}")
    print(f"📊 Training history saved to: {history_path}")
    print("🏁 Training finished.")

if __name__ == "__main__":
    main()
