Here are the results of our experiment

To run experiment, run these commands in order
python generate.py
python part2_train.py
python evaluate.py

# Experiment 1
model:
  block_size: 15
  vocab_size: 16
  n_layer: 1
  n_head: 4
  n_embd: 128
  dropout: 0.1
  bias: false
  tokenizer: "char-level"

train:
  batch_size: 100
  save_dir: "part2_checkpoints/"
  lr: 1e-4
  weight_decay: 0.1
  betas: [0.9, 0.95]
  epochs: 1000
  save_interval: 100
  eval_interval: 1
  seed: 42

data: 
  data_dir: "data"

generate:
  seed: 42
  op: '+'
  prime: 97
  num_samples: 10000

The result of this experiment is documented in part2_checkpoints
Validation Loss after Epoch 1000: 0.0503, Accuracy: 0.9867
--- Test Set Results ---
Average Test Loss: 0.0325
Test Accuracy: 0.9904
Total Correct Predictions: 2879
Total Masked Tokens: 2907

# Experiment 2
model:
  block_size: 15
  vocab_size: 16
  n_layer: 1
  n_head: 4
  n_embd: 128
  dropout: 0.1
  bias: false
  tokenizer: "char-level"

train:
  batch_size: 100
  save_dir: "part2_checkpoints2/"
  lr: 1e-4
  weight_decay: 0.1
  betas: [0.9, 0.95]
  epochs: 1000
  save_interval: 100
  eval_interval: 1
  seed: 42

data: 
  data_dir: "data"

generate:
  seed: 42
  op: '-'
  prime: 97
  num_samples: 10000