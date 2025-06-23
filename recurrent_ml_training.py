from mLN.environment import DynamicSpectrumEnv
from models.recurrent_ml import train_recurrent_maml_ppo
from utils import save_model
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pickle

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from config import Config

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: Config):
    c = cfg.recurrent_ml

    config = {
        "seed": cfg.seed,
        "meta_lr": c.meta_lr,
        "inner_lr": c.inner_lr,
        "inner_steps": c.inner_steps,
        "meta_batch_size": c.meta_batch_size,
        "num_meta_iters": c.num_meta_iters,
        "rollout_length": c.rollout_length,
        "vf_coef": c.vf_coef,
        "ent_coef": c.ent_coef,
        "clip_epsilon": c.clip_epsilon,
        "max_grad_norm": c.max_grad_norm,
        "gamma": c.gamma,
        "lambda_": c.lambda_,
        "use_wandb": c.use_wandb,
        "wandb_project": c.wandb_project,
        "wandb_name": c.wandb_name,
        "log_interval": c.log_interval,
        "eval_interval": c.eval_interval,
        "num_eval_tasks": c.num_eval_tasks,
        "lstm_hidden_dim": c.lstm_hidden_dim,
        "mlp_hidden_dim": c.mlp_hidden_dim,
        "critic_hidden_dim": c.critic_hidden_dim,
    }

    print("Starting Simple Recurrent MAML-PPO with detailed evaluation")
    trained_params, metrics_history = train_recurrent_maml_ppo(config)

    output_dir = HydraConfig.get().runtime.output_dir
    model_dir = os.path.join(output_dir, "models_output")
    os.makedirs(model_dir, exist_ok=True)

    save_model(os.path.join(model_dir, c.save_model_name), trained_params, None)
    with open(os.path.join(model_dir, c.save_history_name), "wb") as f:
        pickle.dump(metrics_history, f)

    print(f"Trained parameters saved to {model_dir}/{c.save_model_name}")
    print(f"Training history saved to {model_dir}/{c.save_history_name}")
    print("Training completed.")

if __name__ == "__main__":
    main()

