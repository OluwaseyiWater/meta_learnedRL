#!/bin/bash

# Set A - Default
python main.py recurrent_attn.meta_lr=2.5e-5 recurrent_attn.inner_lr=1e-4 recurrent_attn.lstm_hidden_dim=128 recurrent_attn.mlp_hidden_dim=128 recurrent_attn.num_attention_heads=4 recurrent_attn.meta_batch_size=16

# Set B - More attention
python main.py recurrent_attn.meta_lr=2e-5 recurrent_attn.inner_lr=5e-5 recurrent_attn.lstm_hidden_dim=128 recurrent_attn.mlp_hidden_dim=256 recurrent_attn.num_attention_heads=8 recurrent_attn.meta_batch_size=16

# Set C - Smaller attention
python main.py recurrent_attn.meta_lr=5e-5 recurrent_attn.inner_lr=2e-4 recurrent_attn.lstm_hidden_dim=64 recurrent_attn.mlp_hidden_dim=64 recurrent_attn.num_attention_heads=2 recurrent_attn.meta_batch_size=8
