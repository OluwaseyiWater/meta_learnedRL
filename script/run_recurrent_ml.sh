#!/bin/bash

# Set A - Default
python main.py recurrent_ml.meta_lr=3e-5 recurrent_ml.inner_lr=1e-4 recurrent_ml.lstm_hidden_dim=64 recurrent_ml.mlp_hidden_dim=128 recurrent_ml.meta_batch_size=16 recurrent_ml.num_meta_iters=500

# Set B - More capacity
python main.py recurrent_ml.meta_lr=3e-5 recurrent_ml.inner_lr=5e-5 recurrent_ml.lstm_hidden_dim=128 recurrent_ml.mlp_hidden_dim=256 recurrent_ml.meta_batch_size=16 recurrent_ml.num_meta_iters=1000

# Set C - Faster meta-learning
python main.py recurrent_ml.meta_lr=1e-4 recurrent_ml.inner_lr=2e-4 recurrent_ml.lstm_hidden_dim=32 recurrent_ml.mlp_hidden_dim=64 recurrent_ml.meta_batch_size=8 recurrent_ml.num_meta_iters=300
