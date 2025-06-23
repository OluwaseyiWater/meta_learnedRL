from dataclasses import dataclass

@dataclass
class RecurrentMLConfig:
    seed: int = 42
    meta_lr: float = 3e-5
    inner_lr: float = 1e-4
    inner_steps: int = 5
    meta_batch_size: int = 16
    num_meta_iters: int = 500
    rollout_length: int = 128
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    clip_epsilon: float = 0.2
    max_grad_norm: float = 1.0
    gamma: float = 0.99
    lambda_: float = 0.95
    use_wandb: bool = False
    wandb_project: str = "maml-spectrum-access"
    wandb_name: str = "simple-recurrent-maml-ppo"
    log_interval: int = 10
    eval_interval: int = 50
    num_eval_tasks: int = 10
    lstm_hidden_dim: int = 64
    mlp_hidden_dim: int = 128
    critic_hidden_dim: int = 128
    save_model_name: str = "ppo_trained_params.pkl"
    save_history_name: str = "ppo_history.pkl"

@dataclass
class Config:
    recurrent_ml: RecurrentMLConfig
