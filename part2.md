Here are the results of our experiment
Seed our experiment 
* change part2config.yaml seed to desired training seed
* change line 53 of generate.py to indicate random seed during generate
* change line 54 of generate.py to indicate number of samples

Grokking experiment 1: model random seed = generate random seed = 42, number of data generated = 10000
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
  eval_interval: 50
  seed: 42

data: 
  data_dir: "data_p97"

Final result: 
Validation Loss after Epoch 1000: 0.1987, Accuracy: 0.9296
--- Test Set Results ---
Average Test Loss: 0.2209
Test Accuracy: 0.9227
Total Correct Predictions: 2675
Total Masked Tokens: 2899


Experiment 2: model random seed = 611, generate random seed = 42

Experiment 3 (training and test curve): model random seed = 19683