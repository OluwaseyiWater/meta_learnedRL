# meta_learnedRL
This is a collection of code for a meta learned reinforcement learning project. The project combines reinforcement learning with meta learning to learn how to learn reinforcement learning policies. The code is written in jax and jumanji.

The project is split into four main components:

* `mLN/environment.py`: This contains the reinforcement learning environment. It is a cellular network environment in which the agent must learn to allocate resources to different users.
* `mLN/agent.py`: This contains the reinforcement learning agent. It is a basic DQN agent.
* `mLN/meta_agent.py`: This contains the meta learning agent. It is a meta learning agent that wraps the reinforcement learning agent.
* `mLN/train.py`: This contains the training code. It trains the meta learning agent to learn how to learn reinforcement learning policies.

* `mLN/`
	+ `environment.py`
