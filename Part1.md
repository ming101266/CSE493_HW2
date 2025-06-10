### Logs of experiments ###

EXP 1: batch_size = 4

did not mask first three tokens

=== TRAINING TEXT ===

['I love machine learning.']

=== DataLoader Batches ===

X batch shape: torch.Size([3, 4])

Y batch shape: torch.Size([3, 4])

Attention Mask batch shape: torch.Size([3, 4])

  Batch 0:

    X (decoded, no pads): '<|endoftext|>I love machine'

    Y (decoded, no pads): 'I love machine learning'

    X (raw tokens): [50256, 40, 1842, 4572]

    Y (raw tokens): [40, 1842, 4572, 4673]

    Attention Mask: [True, True, True, True]

  Batch 1:

    X (decoded, no pads): 'I love machine learning'

    Y (decoded, no pads): ' love machine learning.'

    X (raw tokens): [40, 1842, 4572, 4673]

    Y (raw tokens): [1842, 4572, 4673, 13]

    Attention Mask: [True, True, True, True]

  Batch 2:

    X (decoded, no pads): ' love machine learning.'

    Y (decoded, no pads): ' machine learning.<|endoftext|>'

    X (raw tokens): [1842, 4572, 4673, 13]

    Y (raw tokens): [4572, 4673, 13, 50256]

    Attention Mask: [True, True, True, True]
  


number of parameters: 45.71M

num decayed parameter tensors: 6, with 45,714,432 parameters

num non-decayed parameter tensors: 3, with 2,304 parameters

using fused AdamW: False

Training loss: 0.0000

EXP2:

masked the first three tokens

During inference, the model would only output learning.

This makes sense as the loss of the first three tokens were masked and thus the model only learned to predict "learning."


### Model checkpoint ###

Inside checkpoint folder, currently configured to checkpointing every 100 epochs

### Modifications made ###

Returned optimizer in configure_optimizers

Added option for tokenization
* seems more convenient to have it separately in train and inference.  However, I stored the tokenizer in config so that we can access the config in train while doing inference

Added config.yaml so that I would not have to redeclare my config every time I need to use it

### Challenges ###
1. getting the model to "stop"

I initially forgot about the need for BOS and EOS tokens in the prompts, and wondered how to make the model stop. Then, I decided to pad the start and end of the input training sample with the EOT token, which is what the GPT2 tokenizer uses for both BOS and EOS. 

2. Tokenizer

It was initially tempting to store the full tokenizer object in the config, but it can't be serialized in JSON/YAML. Instead, I stored the tokenizer name ("gpt2") and rebuilt it at runtime using tiktoken.

3. Padding and masking

Because I'm using a causal decoder-only model (GPT), padding doesn't really affect the forward pass as long as all attention is masked correctly. However, I made sure the attention mask was all True in this small example. This will matter more with batching longer inputs, especially with larger block_sizes where padding tokens are likely to appear and need to be masked to avoid attention leaks.
