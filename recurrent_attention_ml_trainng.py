from mLN.environment import DynamicSpectrumEnv
from models.recurrent_attention_ml import  train_recurrent_maml_ppo
from utils import save_model
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pickle
from hydra.core.hydra_config import HydraConfig


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Starting Recurrent MAML-PPO [Optimized Version]")
    print(OmegaConf.to_yaml(cfg.recurrent_attn))

    config = {
        "seed": cfg.seed,
        "meta_lr": cfg.attention_recurrent_ml.meta_lr,
        "inner_lr": cfg.attention_recurrent_ml.inner_lr,
        "inner_steps": cfg.attention_recurrent_ml.inner_steps,
        "meta_batch_size": cfg.attention_recurrent_ml.meta_batch_size,
        "num_meta_iters": cfg.attention_recurrent_ml.num_meta_iters,
        "rollout_length": cfg.attention_recurrent_ml.rollout_length,
        "vf_coef": cfg.attention_recurrent_ml.vf_coef,
        "ent_coef": cfg.attention_recurrent_ml.ent_coef,
        "clip_epsilon": cfg.attention_recurrent_ml.clip_epsilon,
        "max_grad_norm": cfg.attention_recurrent_ml.max_grad_norm,
        "gamma": cfg.attention_recurrent_ml.gamma,
        "lambda_": cfg.attention_recurrent_ml.lambda_,
        "use_wandb": cfg.attention_recurrent_ml.use_wandb,
        "wandb_project": cfg.attention_recurrent_ml.wandb_project,
        "wandb_name": cfg.attention_recurrent_ml.wandb_name,
        "log_interval": cfg.attention_recurrent_ml.log_interval,
        "eval_interval": cfg.attention_recurrent_ml.eval_interval,
        "num_eval_tasks": cfg.attention_recurrent_ml.num_eval_tasks,
        "num_lstm_layers": cfg.attention_recurrent_ml.num_lstm_layers,
        "lstm_hidden_dim": cfg.attention_recurrent_ml.lstm_hidden_dim,
        "mlp_hidden_dim": cfg.attention_recurrent_ml.mlp_hidden_dim,
        "num_attention_heads": cfg.attention_recurrent_ml.num_attention_heads,
    }

    # Train
    trained_params, metrics_history = train_recurrent_maml_ppo(config)

    # Save
    output_dir = HydraConfig.get().runtime.output_dir
    model_dir = os.path.join(output_dir, "models_output")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "recurrent_attn_trained_params.pkl")
    history_path = os.path.join(model_dir, "recurrent_attn_history.pkl")

    save_model(model_path, trained_params, None)
    with open(history_path, "wb") as f:
        pickle.dump(metrics_history, f)

    print(f"✅ Trained model saved to: {model_path}")
    print(f"📊 Metrics history saved to: {history_path}")
    print("🏁 Training finished.")

if __name__ == "__main__":
    main()
