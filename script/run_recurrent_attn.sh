#!/bin/bash

# Set A - Default
python main.py attention_recurrent_ml.meta_lr=2.5e-5 attention_recurrent_ml.inner_lr=1e-4 attention_recurrent_ml.lstm_hidden_dim=128 attention_recurrent_ml.mlp_hidden_dim=128 attention_recurrent_ml.num_attention_heads=4 attention_recurrent_ml.meta_batch_size=16

# Set B - More attention
python main.py attention_recurrent_ml.meta_lr=2e-5 attention_recurrent_ml.inner_lr=5e-5 attention_recurrent_ml.lstm_hidden_dim=128 attention_recurrent_ml.mlp_hidden_dim=256 attention_recurrent_ml.num_attention_heads=8 attention_recurrent_ml.meta_batch_size=16

# Set C - Smaller attention
python main.py attention_recurrent_ml.meta_lr=5e-5 attention_recurrent_ml.inner_lr=2e-4 attention_recurrent_ml.lstm_hidden_dim=64 attention_recurrent_ml.mlp_hidden_dim=64 attention_recurrent_ml.num_attention_heads=2 attention_recurrent_ml.meta_batch_size=8
