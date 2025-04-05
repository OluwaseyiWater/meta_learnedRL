import pickle
import jax.numpy as jnp
import wandb
from graphviz import Digraph
import tempfile
import os


def save_model(path, params, st):
    with open(path, 'wb') as f:
        pickle.dump({'params': params, 'st': st}, f)
    print("Parameters saved to ", path)
    
def load_model(path):
  with open(path, 'rb') as f:
    data = pickle.load(f)
  params = data['params']
  st = data['st']
  print("Parameters loaded from ", path)
  return params, st

    
def flatten_state(state):
    return jnp.concatenate([
        state.channel_gains.flatten().astype(jnp.float32),
        state.interference_map.flatten().astype(jnp.float32),
        state.qos_metrics.flatten().astype(jnp.float32),
        state.spectrum_alloc.flatten().astype(jnp.float32),
        state.tx_power.flatten().astype(jnp.float32)
    ])
    
    
def log_network_architecture_to_wandb(policy_params, value_params, num_bs, num_bands, num_power_levels):
    """
    Create a visualization of the network architecture and log it to W&B
    
    Args:
        policy_params: Policy network parameters
        value_params: Value network parameters
        num_bs: Number of base stations
        num_bands: Number of frequency bands
        num_power_levels: Number of power levels
    """
    
    # Create a temporary directory for the visualization
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create policy network visualization
        dot = Digraph(comment='Recurrent Policy Network')
        
        # Add nodes
        dot.node('input', 'Input Observation')
        dot.node('flatten', 'Flatten')
        dot.node('linear1', f'Linear({policy_params[0].shape[1]})')
        dot.node('relu1', 'ReLU')
        dot.node('lstm', f'LSTM({policy_params[2].shape[0]})')
        dot.node('output', f'Linear({num_bs * num_bands * num_power_levels})')
        dot.node('reshape', f'Reshape({num_bs * num_bands}, {num_power_levels})')
        
        # Add edges
        dot.edge('input', 'flatten')
        dot.edge('flatten', 'linear1')
        dot.edge('linear1', 'relu1')
        dot.edge('relu1', 'lstm')
        dot.edge('lstm', 'output')
        dot.edge('output', 'reshape')
        
        # Save visualization
        policy_viz_path = os.path.join(tmpdirname, 'policy_network.png')
        dot.render(outfile=policy_viz_path, format='png')
        
        # Create value network visualization
        dot = Digraph(comment='Value Network')
        
        # Add nodes
        dot.node('input', 'Input Observation')
        dot.node('flatten', 'Flatten')
        dot.node('linear1', f'Linear({value_params[0].shape[1]})')
        dot.node('relu1', 'ReLU')
        
        # Add residual blocks
        for i in range(3):  # Assuming 3 residual blocks
            dot.node(f'resblock{i}', f'Residual Block {i+1}')
            if i == 0:
                dot.edge('relu1', f'resblock{i}')
            else:
                dot.edge(f'resblock{i-1}', f'resblock{i}')
        
        dot.node('output', 'Linear(1)')
        dot.edge(f'resblock2', 'output')
        
        # Save visualization
        value_viz_path = os.path.join(tmpdirname, 'value_network.png')
        dot.render(outfile=value_viz_path, format='png')
        
        # Log to W&B
        try:
            wandb.log({
                "policy_network_architecture": wandb.Image(policy_viz_path),
                "value_network_architecture": wandb.Image(value_viz_path)
            })
        except Exception as e:
            print(f"Error logging network architecture to W&B: {e}")