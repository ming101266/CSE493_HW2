Here are the results of our experiment

To run experiment, run these commands in order
python generate.py
python part2_train.py
python evaluate.py

# Data Generation
The number of data generated and location should be specified in part2config.yaml 

generate:
  seed: 42
  op: '/'
  prime: 97
  num_samples: 10000
  data_dir: "ablation1/"

The training, validation, testing split is 50, 25, 25 and each file (test.txt, train.txt, val.txt) can be found in the folder for each experiment. 

# Addition and Subtraction Experiments

The results of these experiments are within each folder. Due to the size of each model, it was impossible to include the folder for each so a summary of each result in included at the bottom of this file. The 3 random seeds chosen were 42, 611, and 19683.

# Grokking

# Ablation

# Part 2: Algorithmic Tasks # 

We generated a dataset of 10,000 expressions for modular arithmetic. To create each sample, we randomly selected two integers between 0 and p−1, then applied a chosen operator to compute the result modulo p. Specifically, we calculated c=(a op b)mod p, and formatted the sample as a string in the form a{op}b=c. This process was repeated until we had 10,000 samples. Afterward, we randomly shuffled the dataset and split it into 3,000 examples for training, 3,500 for validation, and 3,500 for testing.

The training and test curves and a zip of final model checkpoint for each of the operators can be found in our github link. 

## To Add or Not to Add: That Is the Question

When a transformer model is trained on modular addition (e.g., `a + b mod 97 → c`), it develops internal representations that associate pairs of input numbers with a specific output. These associations aren’t based on symbolic understanding of `+` or modular arithmetic rules, but instead arise from statistical pattern matching and correlation building.

So what happens when we present the same model with a subtraction prompt, like `a - b mod 97 → ?`, even though it was never trained on subtraction?

### The Core Mechanism

The model, having only seen `a + b` during training, builds strong correlations between:
- the pair of inputs `(a, b)`
- and the result `a + b mod 97`

It essentially learns:
- “When I see input tokens `a+b=`, I expect output token `c`” , where `c = a + b mod 97` during training

Thus, when it it observes tokens 'a' and 'b', it learns to predict 'c'. It learns that these three numbers are closely related to each other. 

Thus, when the model sees a subtraction prompt, e.g., `45 - 12 → ?`, since it has never encounted the operator `-`, it guesses between the two groups which are associated with 12 and 45, 

- `45 + 12 mod 97 = 57`
- and
- `12 + 33 mod 97 = 45`

Since both groups appear as valid, strongly associated outcomes for the same input pair `(a, b)`, the model faces an ambiguous internal representation. 
Then, it essentially flips a coin between the two possible learned correlations.

This results in:
Approximately 50% test accuracy on subtraction prompts even though the model was never trained on subtraction, which matches our observation. 

### Evidence: Model Is Not Guessing Randomly

It’s important to note: this isn’t uniform random guessing over 97 outputs. That would yield ~1% accuracy.

Instead, the model is guessing between two meaningful candidates and choosing one based on incomplete or misinterpreted operator signals.

This means the model has partially internalized the group structure of mod 97, but lacks a symbolic or semantic understanding of what the operator token `-` actually means,

Thus, in the absence of explicit subtraction training, the model reduces the decision to:

> **To add or not to add? That is the question.**

It knows the input pair is associated with something, but lacking clear operator context, it guesses between plausible meanings. That ambiguity produces the mysterious ~50% accuracy on subtraction: a sign not of confusion, but of partial understanding constrained by training.

### Further Suggestions ###
- **Evaluate Few-Shot Generalization from Addition to Subtraction**

  Investigate whether a model trained exclusively on modular addition can generalize to modular subtraction through 1-shot or few-shot learning. Specifically, test whether providing the model with a very small number of subtraction examples (e.g., 1–5 examples) enables it to correctly interpret and generalize the meaning of the `-` operator. If successful, this would suggest that the model’s internal representations of modular arithmetic are structurally rich enough to support rapid task transfer across operations that share algebraic foundations.

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

