import torch
from torch.utils.data import Dataset

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
  
class TextDataset(Dataset):
    def __init__(self, tokenized_expressions):
        self.tokenized_expressions = tokenized_expressions

    def __len__(self):
        return len(self.tokenized_expressions)

    def __getitem__(self, idx):
        return self.tokenized_expressions[idx]

def custom_collate_fn(batch, pad_token_id, block_size, tokenizer):
    max_seq_len_in_batch = max(len(expr_tokens) for expr_tokens in batch)
    actual_block_size = max(block_size, max_seq_len_in_batch)

    padded_x_batch = []
    padded_y_batch = []
    attention_mask_batch = []
    loss_mask_batch = []
    
    eq_token_id = tokenizer.stoi['=']

    for expr_tokens in batch:
        x_seq = expr_tokens[:-1]
        y_seq = expr_tokens[1:]

        padding_len = actual_block_size - len(x_seq)
        
        padded_x = x_seq + [pad_token_id] * padding_len
        padded_y = y_seq + [pad_token_id] * padding_len
        
        attention_mask = [True] * len(x_seq) + [False] * padding_len
        
        current_loss_mask = [False] * actual_block_size
        
        eq_idx_in_expr = -1
        try:
            eq_idx_in_expr = expr_tokens.index(eq_token_id)
        except ValueError:
            pass
        
        if eq_idx_in_expr != -1:
            for i in range(eq_idx_in_expr, len(y_seq)):
                current_loss_mask[i] = True

        padded_x_batch.append(padded_x)
        padded_y_batch.append(padded_y)
        attention_mask_batch.append(attention_mask)
        loss_mask_batch.append(current_loss_mask)

    return (
        torch.tensor(padded_x_batch, dtype=torch.long),
        torch.tensor(padded_y_batch, dtype=torch.long),
        torch.tensor(attention_mask_batch, dtype=torch.bool),
        torch.tensor(loss_mask_batch, dtype=torch.bool)
    )