import torch
from torch.utils.data import Dataset
import re

# -------- Custom Character Tokenizer --------
class CharTokenizer:
    def __init__(self, text_corpus):
        # Build vocabulary from the entire text corpus
        # We need to handle special tokens: equals, plus, minus, divide
        # And the digits 0-9, and potentially spaces if they are part of the numbers
        # The equation format is "a+b=c", "a-b=c", "a/b=c"
        # So, the characters involved are digits, '+', '-', '/', '='
        # We also need a special EOT token and a PAD token.
        
        unique_chars = sorted(list(set(char for line in text_corpus for char in line.strip())))
        
        # Add special tokens
        self.eot_token = len(unique_chars) # End of Text
        self.pad_token = self.eot_token + 1 # Padding
        
        # Create mappings
        self.stoi = {char: i for i, char in enumerate(unique_chars)}
        self.itos = {i: char for i, char in enumerate(unique_chars)}
        
        self.stoi['<EOT>'] = self.eot_token
        self.itos[self.eot_token] = '<EOT>'
        self.stoi['<PAD>'] = self.pad_token
        self.itos[self.pad_token] = '<PAD>'
        
        self.vocab_size = len(self.stoi)

    def encode(self, text):
        return [self.stoi[char] for char in text]

    def decode(self, tokens):
        return "".join([self.itos[token] for token in tokens if token in self.itos])

    @property
    def max_token_value(self):
        return self.vocab_size - 1

class NumberTokenizer:
    def __init__(self, max_number=199):
        # Numbers: 0 to max_number
        number_tokens = [str(i) for i in range(max_number + 1)]
        operator_tokens = ['+', '-', '/', '=']

        all_tokens = number_tokens + operator_tokens
        all_tokens.sort(key=lambda x: (len(x), x))  # optional: sort by length then lexically

        # Add special tokens
        self.eot_token = len(all_tokens)
        self.pad_token = self.eot_token + 1

        self.stoi = {tok: i for i, tok in enumerate(all_tokens)}
        self.itos = {i: tok for tok, i in self.stoi.items()}

        self.stoi['<EOT>'] = self.eot_token
        self.itos[self.eot_token] = '<EOT>'
        self.stoi['<PAD>'] = self.pad_token
        self.itos[self.pad_token] = '<PAD>'

        self.vocab_size = len(self.stoi)

    def encode(self, text):
        # Tokenize into numbers and operators
        tokens = re.findall(r'\d+|[+\-=/]', text.strip())
        return [self.stoi[tok] for tok in tokens] + [self.stoi['<EOT>']]

    def decode(self, token_ids):
        tokens = [self.itos[tok] for tok in token_ids if tok in self.itos and self.itos[tok] != '<PAD>']
        return "".join(tokens)

    @property
    def max_token_value(self):
        return self.vocab_size - 1

class TextDataset(Dataset):
    def __init__(self, tokenized_expressions):
        self.tokenized_expressions = tokenized_expressions

    def __len__(self):
        return len(self.tokenized_expressions)

    def __getitem__(self, idx):
        return self.tokenized_expressions[idx]
def custom_collate_fn(batch, pad_token_id, block_size, tokenizer):
    padded_x_batch = []
    padded_y_batch = []
    attention_mask_batch = []
    loss_mask_batch = []

    for expr_tokens in batch:
        try:
            eq_idx = expr_tokens.index(tokenizer.stoi['='])
        except ValueError:
            eq_idx = -1  # fallback if no '='

        # x: <EOT> + lhs including '='
        x_seq = expr_tokens[:eq_idx + 1]

        # y: full expression (exclude <EOT>)
        y_seq = expr_tokens[1:-1]

        # Pad to block_size
        x_padding_len = block_size - len(x_seq)
        y_padding_len = block_size - len(y_seq)

        padded_x = [pad_token_id] * x_padding_len + x_seq
        padded_y = y_seq + [pad_token_id] * y_padding_len

        attention_mask = [False] * x_padding_len + [True] * len(x_seq)

        # Loss mask: only after '=' in y_seq
        current_loss_mask = [False] * eq_idx + [True] * (len(y_seq) - eq_idx) + [False] *(y_padding_len)
        padded_x_batch.append(padded_x)
        padded_y_batch.append(padded_y)
        attention_mask_batch.append(attention_mask)
        loss_mask_batch.append(current_loss_mask)

    return (
        torch.tensor(padded_x_batch, dtype=torch.long),
        torch.tensor(padded_y_batch, dtype=torch.long),
        torch.tensor(attention_mask_batch, dtype=torch.bool),
        torch.tensor(loss_mask_batch, dtype=torch.bool),
    )
