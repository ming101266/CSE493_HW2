import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import tiktoken
from model import GPT, GPTConfig

import os

# -------- Dataset --------

class TextDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        # We need to ensure that there's always enough tokens for an input (block_size) and a target (block_size)
        # So the last possible starting index for x is len(self.tokens) - self.block_size
        return max(0, len(self.tokens) - self.block_size)

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + 1 + self.block_size]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
    
def custom_collate_fn(batch, pad_token_id, block_size):
    """
    Custom collate function to pad sequences in a batch to the maximum length.
    """
    x_batch, y_batch = zip(*batch)
    
    max_len = max(len(seq) for seq in x_batch)
    max_len = max(block_size, max_len)
    
    padded_x_batch = []
    padded_y_batch = []
    attention_mask_batch = []
    
    for x, y in zip(x_batch, y_batch):
        padding_len = max_len - len(x)
        # Pad with pad_token_id
        padded_x_batch.append(torch.cat((x, torch.full((padding_len,), pad_token_id, dtype=torch.long))))
        padded_y_batch.append(torch.cat((y, torch.full((padding_len,), pad_token_id, dtype=torch.long))))
        # Create attention mask: 1 for original tokens, 0 for padding
        attention_mask_batch.append(padded_x_batch[-1] != pad_token_id)

    return torch.stack(padded_x_batch), torch.stack(padded_y_batch), torch.stack(attention_mask_batch)

# -------- Training --------

def train(
    data_path="input.txt",
    epochs=501,
    batch_size=12,
    block_size=12,
    lr=3e-4,
    weight_decay=0.1,
    betas=(0.9, 0.95),
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_dir="checkpoints",
):
    # Get tokenizer and define pad token
    tokenizer = tiktoken.get_encoding("gpt2")
    pad_token_id = tokenizer.max_token_value + 1

    # 1. Load text data
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.readlines()
    print("### TRAINING TEXT ###")
    print(text)
    print("######################")

    # 2. Dataset and DataLoader
    tokens = []
    for line in text:
        line = line.strip()
        if line:
            line_tokens = [tokenizer.eot_token] + tokenizer.encode(line) + [tokenizer.eot_token]
            tokens.extend(line_tokens)
    
    # avoid empty dataloader
    if len(tokens) < block_size + 1:
        tokens.extend([pad_token_id] * (block_size + 1 - len(tokens)))
    
    dataset = TextDataset(tokens, block_size)
    # Use a custom collate_fn to handle padding and create attention mask for each batch
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: custom_collate_fn(b, pad_token_id, batch_size))

    # All batches (after padding and masking by collate_fn)
    print("\n=== DataLoader Batch Example ===")
    for i, (x_batch, y_batch, attention_mask_batch) in enumerate(dataloader):
        print(f"X batch shape: {x_batch.shape}")
        print(f"Y batch shape: {y_batch.shape}")
        print(f"Attention Mask batch shape: {attention_mask_batch.shape}")
        for batch_idx in range(x_batch.size(0)):
            x_tokens = x_batch[batch_idx].tolist()
            y_tokens = y_batch[batch_idx].tolist()
            mask = attention_mask_batch[batch_idx].tolist()
            # Filter out pad_token_id for display purposes
            x_text = tokenizer.decode([token for token in x_tokens if token != pad_token_id])
            y_text = tokenizer.decode([token for token in y_tokens if token != pad_token_id])
            print(f"  Batch {batch_idx}:")
            print(f"    X (decoded, no pads): '{x_text}'")
            print(f"    Y (decoded, no pads): '{y_text}'")
            print(f"    X (raw tokens): {x_tokens}")
            print(f"    Y (raw tokens): {y_tokens}")
            print(f"    Attention Mask: {mask}")
    print("==============================================\n")


    # 3. Model config and instantiate model
    config = GPTConfig(
        block_size=block_size,
        vocab_size=50304,
        n_layer=1,
        n_head=1,
        n_embd=768,
        dropout=0.1,
        bias=False
    )
    model = GPT(config).to(device)
    config.pad_token_id = pad_token_id

    # 4. Optimizer
    optimizer = model.configure_optimizers(weight_decay, lr, betas, device_type=device)

    # 5. Training loop
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        for i, (x, y, attention_mask) in enumerate(dataloader): # Unpack attention_mask
            x, y, attention_mask = x.to(device), y.to(device), attention_mask.to(device)

            optimizer.zero_grad()
            logits = model(x, attention_mask=attention_mask) 
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=pad_token_id
            )
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 100 == 0:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch{epoch + 1}.pt"))
            print(f"Saved checkpoint for epoch {epoch + 1}")

if __name__ == "__main__":
    train()