from mLN.environment import DynamicSpectrumEnv
from models.recurrent_attention_ml import make_robust_recurrent_networks as make_recurrent_networks, train_recurrent_maml_ppo
from utils import save_model, flatten_state
import jax
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pickle

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Convert cfg.recurrent_ml to flat dict
    flat_cfg = OmegaConf.to_container(cfg.recurrent_ml, resolve=True)
    flat_cfg["seed"] = cfg.seed

    env = hydra.utils.instantiate(cfg.env)
    flat_cfg["env"] = env

    key = jax.random.PRNGKey(cfg.seed)
    key, p_key, v_key, t_key = jax.random.split(key, 4)
    flat_cfg["key"] = t_key

    dummy_state = env.observation_spec().generate_value()
    dummy_obs = flatten_state(dummy_state)[None, :]

    num_actions = env.num_bs * env.num_bands
    num_action_values = env.num_power_levels
    num_outputs = num_actions * num_action_values

    actor, critic = make_recurrent_networks(num_outputs, flat_cfg)
    actor_params = actor.init(p_key, dummy_obs, None)
    critic_params = critic.init(v_key, dummy_obs.squeeze(0))
    flat_cfg["policy_params"] = actor_params
    flat_cfg["value_params"] = critic_params
    flat_cfg["policy_apply"] = actor.apply
    flat_cfg["value_apply"] = critic.apply
    flat_cfg["obs_dim"] = dummy_obs.shape[-1]

    trained_params, history = train_recurrent_maml_ppo(flat_cfg)

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    model_dir = os.path.join(output_dir, "models_output")
    os.makedirs(model_dir, exist_ok=True)

    save_model(os.path.join(model_dir, "trained_params.pkl"), trained_params, None)
    with open(os.path.join(model_dir, "history.pkl"), 'wb') as f:
        pickle.dump(history, f)

    print(f"Trained parameters saved to {model_dir}/trained_params.pkl")
    print(f"Training history saved to {model_dir}/history.pkl")
    print(f"Number of base stations: {env.num_bs}")
    print(f"Number of bands: {env.num_bands}")
    print(f"Number of power levels: {env.num_power_levels}")
    print("Training completed.")

if __name__ == "__main__":
    main()

