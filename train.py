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
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + 1 + self.block_size]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# -------- Training --------

def train(
    data_path="input.txt",
    epochs=201,
    batch_size=2,
    block_size=2,
    lr=3e-4,
    weight_decay=0.1,
    betas=(0.9, 0.95),
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_dir="checkpoints",
):

    # 1. Load text data
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.readlines()
    print("### TRAINING TEXT ###")
    print(text)
    print("######################")

    # 2. Dataset and DataLoader
    tokenizer = tiktoken.get_encoding("gpt2")
    bos_token = tokenizer.eot_token
    tokens = []
    for line in text:
        line = line.strip()
        if line:
            line_tokens = [bos_token] + tokenizer.encode(line) + [bos_token]
            tokens.extend(line_tokens)

    dataset = TextDataset(tokens, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for i, (x, y) in enumerate(dataloader):
        for batch_idx in range(x.size(0)):
            x_tokens = x[batch_idx].tolist()  # list of ints
            y_tokens = y[batch_idx].tolist()
            x_text = tokenizer.decode(x_tokens)
            y_text = tokenizer.decode(y_tokens)
            print(f"X: {x_text}")
            print(f"Y: {y_text}\n")


    # 3. Model config and instantiate model
    config = GPTConfig(
        block_size=block_size,
        vocab_size=50304,
        n_layer=1,
        n_head=1,
        n_embd=768,
        dropout=0.1,
        bias=False,
    )
    model = GPT(config).to(device)

    # 4. Optimizer
    optimizer = model.configure_optimizers(weight_decay, lr, betas, device_type=device)

    # 5. Training loop
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)  # shape (batch, seq_len, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 100 == 0:
                print(f"Step {i}, loss = {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        # Save checkpoint
        if epoch % 100 == 0:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch{epoch}.pt"))
            print(f"Saved checkpoint for epoch {epoch}")

if __name__ == "__main__":
    train()
