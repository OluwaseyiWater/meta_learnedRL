import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pickle
from models.maml import train_maml
from utils import save_model 


@hydra.main(config_path="conf/maml", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Starting MAML training...")
    print(OmegaConf.to_yaml(cfg))
    maml_config = OmegaConf.to_container(cfg, resolve=True)
    maml_config['seed'] = cfg.seed
    maml_config['ent_coef_start'] = 0.02
    maml_config['ent_coef_end'] = 0.001
    maml_config['ent_coef_decay_steps'] = cfg.num_meta_iters // 2
    trained_params, history = train_maml(maml_config)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    model_dir = os.path.join(output_dir, "models_output")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "maml_trained_params.pkl")
    history_path = os.path.join(model_dir, "maml_history.pkl")

    save_model(model_path, trained_params, None)
    
    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    print(f"✅ Trained model saved to: {model_path}")
    print(f"📊 Training history saved to: {history_path}")
    print("🏁 Training finished.")


if __name__ == "__main__":
    main()
