from mLN.environment import DynamicSpectrumEnv
from models.maml import train_maml
from utils import save_model
import hydra
from omegaconf import DictConfig, OmegaConf
import os

# Hydra decorator to load config from a YAML file
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Print the resolved configuration for debugging
    print(OmegaConf.to_yaml(cfg))

    # Instantiate the environment using Hydra's instantiate utility
    env = hydra.utils.instantiate(cfg.env)

    # Train MAML with parameters from the config
    trained_params = train_maml(
        env,
        num_meta_tasks=cfg.training.num_meta_tasks,
        inner_steps=cfg.training.inner_steps,
        meta_lr=cfg.training.meta_lr,
        inner_lr=cfg.training.inner_lr
    )

    # Get the output directory managed by Hydra
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Save the trained parameters
    save_path = os.path.join(model_dir, "trained_params.pkl")
    save_model(save_path, trained_params, None)
    print(f"Trained parameters saved to {save_path}")

    # Print environment details
    print(f"Number of base stations: {env.num_bs}")
    print(f"Number of bands: {env.num_bands}")
    print(f"Number of power levels: {env.num_power_levels}")
    print("Training completed.")


if __name__ == "__main__":
    main()