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

Validation Loss after Epoch 1000: 0.0936, Accuracy: 0.9660
--- Test Set Results ---
Average Test Loss: 0.1046
Test Accuracy: 0.9645
Total Correct Predictions: 2802
Total Masked Tokens: 2905

# Experiment 3
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
  save_dir: "part2_checkpoints3/"
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
  prime: 113
  num_samples: 10000

Validation Loss after Epoch 1000: 0.1025, Accuracy: 0.9662
--- Test Set Results ---
Average Test Loss: 0.0946
Test Accuracy: 0.9665
Total Correct Predictions: 2939
Total Masked Tokens: 3041

# Experiment 4
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
  lr: 1e-4
  save_dir: "part2_checkpoints4/"
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
  prime: 113
  num_samples: 10000

# Experiment 5
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
  seed: 611
  op: '+'
  prime: 97
  num_samples: 10000
Validation Loss after Epoch 1000: 0.1674, Accuracy: 0.9414
--- Test Set Results ---
Average Test Loss: 0.1512
Test Accuracy: 0.9451
Total Correct Predictions: 2859
Total Masked Tokens: 3025

# Experiment 6
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
  save_dir: "part2_checkpoints4/"
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
  seed: 611
  op: '-'
  prime: 97
  num_samples: 10000
Validation Loss after Epoch 1000: 0.1063, Accuracy: 0.9640
--- Test Set Results ---
Average Test Loss: 0.0978
Test Accuracy: 0.9697
Total Correct Predictions: 2814
Total Masked Tokens: 2902

# Experiment 7
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
  save_dir: "part2_checkpoints7/"
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
  seed: 611
  op: '+'
  prime: 113
  num_samples: 10000
Validation Loss after Epoch 1000: 0.1041, Accuracy: 0.9688

--- Test Set Results ---
Average Test Loss: 0.1155
Test Accuracy: 0.9601
Total Correct Predictions: 2914
Total Masked Tokens: 3035

# Experiment 8
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
  save_dir: "part2_checkpoints8/"
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
  seed: 611
  op: '-'
  prime: 113
  num_samples: 10000
Validation Loss after Epoch 1000: 0.1835, Accuracy: 0.9347
--- Test Set Results ---
Average Test Loss: 0.1922
Test Accuracy: 0.9325
Total Correct Predictions: 2831
Total Masked Tokens: 3036

# Experiment 9

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
  save_dir: "part2_checkpoints9/"
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
  seed: 19683
  op: '+'
  prime: 97
  num_samples: 10000

# Experiment 10
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
  save_dir: "part2_checkpoints10/"
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
  seed: 19683
  op: '-'
  prime: 97
  num_samples: 10000

# Experiment 11
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
  save_dir: "part2_checkpoints11/"
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
  seed: 19683
  op: '+'
  prime: 113
  num_samples: 10000

# Experiment 12
model:
  block_size: 15
  vocab_size: 16"grokk/"
  bias: false
  tokenizer: "char-level"

train:
  batch_size: 100
  save_dir: "part2_checkpoints12/"
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
  seed: 19683
  op: '-'
  prime: 113
  num_samples: 10000

 Experiment 1
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


# Experiment 13
model:
  block_size: 15
  vocab_size: 16
  n_layer: 2
  n_head: 4
  n_embd: 128
  dropout: 0.1
  bias: false
  tokenizer: "char-level"

train:
  batch_size: 100
  save_dir: "part2_checkpoints13/"
  lr: 1e-4
  weight_decay: 0.1
  betas: [0.9, 0.95]
  epochs: 1000
  save_interval: 100
  eval_interval: 1
  seed: 42

data: 
  data_dir: "part2_checkpoints13/"

generate:
  seed: 42
  op: '-'
  prime: 97
  num_samples: 10000
  data_dir: "part2_checkpoints13/"

# Experiment 14
model:
  block_size: 15
  vocab_size: 16
  n_layer: 2
  n_head: 4
  n_embd: 128
  dropout: 0.1
  bias: false
  tokenizer: "char-level"

train:
  batch_size: 100
  save_dir: "part2_checkpoints14/"
  lr: 1e-4
  weight_decay: 0.1
  betas: [0.9, 0.95]
  epochs: 1000
  save_interval: 100
  eval_interval: 1
  seed: 42

data: 
  data_dir: "part2_checkpoints14/"

generate:
  seed: 42
  op: '-'
  prime: 97
  num_samples: 10000
  save_dir: "part2_checkpoints14/"

# Experiment 15
model:
  block_size: 15
  vocab_size: 16
  n_layer: 2
  n_head: 4
  n_embd: 128
  dropout: 0.1
  bias: false
  tokenizer: "char-level"

train:
  batch_size: 100
  save_dir: "part2_checkpoints15/"
  lr: 1e-4
  weight_decay: 0.1
  betas: [0.9, 0.95]
  epochs: 1000
  save_interval: 100
  eval_interval: 1
  seed: 42

data: 
  data_dir: "part2_checkpoints15/"

generate:
  seed: 42
  op: '+'
  prime: 113
  num_samples: 10000
  data_dir: "part2_checkpoints15/"


# Experiment 4
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
  lr: 1e-4
  save_dir: "part2_checkpoints4/"
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
  prime: 113
  num_samples: 10000

# Experiment 5
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
  seed: 611
  op: '+'
  prime: 97
  num_samples: 10000
Validation Loss after Epoch 1000: 0.1674, Accuracy: 0.9414
--- Test Set Results ---
Average Test Loss: 0.1512
Test Accuracy: 0.9451
Total Correct Predictions: 2859
Total Masked Tokens: 3025

# Experiment 6
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
  save_dir: "part2_checkpoints4/"
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
  seed: 611
  op: '-'
  prime: 97
  num_samples: 10000
Validation Loss after Epoch 1000: 0.1063, Accuracy: 0.9640
--- Test Set Results ---
Average Test Loss: 0.0978
Test Accuracy: 0.9697
Total Correct Predictions: 2814
Total Masked Tokens: 2902

# Experiment 7
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
  save_dir: "part2_checkpoints7/"
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
  seed: 611
  op: '+'
  prime: 113
  num_samples: 10000
Validation Loss after Epoch 1000: 0.1041, Accuracy: 0.9688

--- Test Set Results ---
Average Test Loss: 0.1155
Test Accuracy: 0.9601
Total Correct Predictions: 2914
Total Masked Tokens: 3035

# Experiment 8
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
  save_dir: "part2_checkpoints8/"
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
  seed: 611
  op: '-'
  prime: 113
  num_samples: 10000
Validation Loss after Epoch 1000: 0.1835, Accuracy: 0.9347
--- Test Set Results ---
Average Test Loss: 0.1922
Test Accuracy: 0.9325
Total Correct Predictions: 2831
Total Masked Tokens: 3036

# Experiment 9

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
  save_dir: "part2_checkpoints9/"
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
  seed: 19683
  op: '+'
  prime: 97
  num_samples: 10000

# Experiment 10
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
  save_dir: "part2_checkpoints10/"
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
  seed: 19683
  op: '-'
  prime: 97
  num_samples: 10000

# Experiment 11
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
  save_dir: "part2_checkpoints11/"
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
  seed: 19683
  op: '+'
  prime: 113
  num_samples: 10000

# Experiment 12
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
  save_dir: "part2_checkpoints12/"
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
  seed: 19683
  op: '-'
  prime: 113
  num_samples: 10000

