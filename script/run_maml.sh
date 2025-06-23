#!/bin/bash

# Set A - Baseline
python main.py maml.meta_lr=3e-4 maml.inner_policy_lr=1e-3 maml.inner_value_lr=1e-3 maml.meta_batch_size=16 maml.inner_steps=5

# Set B - Faster adaptation
python main.py maml.meta_lr=1e-4 maml.inner_policy_lr=2e-3 maml.inner_value_lr=2e-3 maml.meta_batch_size=8 maml.inner_steps=3

# Set C - More stable meta-update
python main.py maml.meta_lr=5e-5 maml.inner_policy_lr=5e-4 maml.inner_value_lr=5e-4 maml.meta_batch_size=32 maml.inner_steps=5
