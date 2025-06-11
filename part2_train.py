import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import yaml
import os
from datetime import datetime
from tqdm import tqdm # Import tqdm for progress bars
from transformers import get_cosine_schedule_with_warmup # For learning rate scheduling
import random
import numpy as np
import matplotlib.pyplot as plt

from model import GPT, GPTConfig
from utils import CharTokenizer, TextDataset, custom_collate_fn

# -------- Dataset --------

def prepare_dataset_tokens(filepath, tokenizer):
    list_of_tokenized_expressions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                tokenized_line = [tokenizer.eot_token] + tokenizer.encode(line) + [tokenizer.eot_token]
                list_of_tokenized_expressions.append(tokenized_line)
    return list_of_tokenized_expressions

# -------- Training Function --------
def train():
    with open("part2config.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)

    model_cfg = cfg_dict["model"]
    train_cfg = cfg_dict["train"]
    data_cfg = cfg_dict["data"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    all_text = []
    data_dir_path = data_cfg.get("data_dir", "data_p97")
    for split_file in ["train.txt", "val.txt", "test.txt"]:
        file_path = os.path.join(data_dir_path, split_file)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                all_text.extend(f.readlines())
    
    if not all_text:
        raise RuntimeError(f"No text found in {data_dir_path}. Cannot initialize tokenizer. Please ensure data is generated and path is correct.")

    tokenizer = CharTokenizer(text_corpus=all_text)
    pad_token_id = tokenizer.pad_token
    print(f"Vocab Size: {tokenizer.vocab_size}")
    print(f"Tokenizer stoi: {tokenizer.stoi}")
    print(f"Pad Token ID: {pad_token_id}")

    config = GPTConfig(**model_cfg)
    config.vocab_size = tokenizer.vocab_size
    seed = train_cfg["seed"]

    # Seeding
    random.seed(seed)                  # Python RNG
    np.random.seed(seed)               # NumPy RNG
    torch.manual_seed(seed)            # CPU RNG
    torch.cuda.manual_seed(seed)       # GPU RNG
    torch.cuda.manual_seed_all(seed)   # All GPUs


    model = GPT(config).to(device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = model.configure_optimizers(
        train_cfg["weight_decay"],
        float(train_cfg["lr"]),
        tuple(train_cfg["betas"]),
        device_type=device
    )

    train_expressions = prepare_dataset_tokens(os.path.join(data_dir_path, "train.txt"), tokenizer)
    val_expressions = prepare_dataset_tokens(os.path.join(data_dir_path, "val.txt"), tokenizer)

    train_dataset = TextDataset(train_expressions)
    val_dataset = TextDataset(val_expressions)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: custom_collate_fn(batch, pad_token_id, config.block_size, tokenizer),
        num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=lambda batch: custom_collate_fn(batch, pad_token_id, config.block_size, tokenizer),
        num_workers=4
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Debug print for DataLoader Batches
    print("\n=== DataLoader Batches Example (Train) ===\n")
    for i, (x, y, attention_mask, loss_mask) in enumerate(train_dataloader):
        print(f"Batch {i}:")
        print(f"  X batch shape: {x.shape}")
        print(f"  Y batch shape: {y.shape}")
        print(f"  Attention Mask batch shape: {attention_mask.shape}")
        print(f"  Loss Mask batch shape: {loss_mask.shape}")
        for j in range(min(4, x.shape[0])):
            print(f"    Sample {j}:")
            print(f"      X (decoded, no pads): {tokenizer.decode(x[j][attention_mask[j]].tolist())}")
            print(f"      Y (decoded, no pads): {tokenizer.decode(y[j][attention_mask[j]].tolist())}")
            print(f"      X (raw tokens): {x[j].tolist()}")
            print(f"      Y (raw tokens): {y[j].tolist()}")
            print(f"      Attention Mask: {attention_mask[j].tolist()}")
            print(f"      Loss Mask: {loss_mask[j].tolist()}")
        break # only showing first 1
    print("==============================================\n")

    # Learning Rate Scheduler
    num_training_steps = train_cfg["epochs"] * len(train_dataloader)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_cfg.get("warmup_steps", 0),
        num_training_steps=num_training_steps
    )

    # Automatic Mixed Precision (AMP) Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.get("use_amp", False))

    model.train()
    global_step = 0
    start_time = datetime.now()
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for epoch in range(train_cfg["epochs"]):
        print(f"Epoch {epoch + 1}/{train_cfg['epochs']}")
        total_loss = 0
        total_correct_predictions = 0
        total_masked_tokens = 0

        # Use tqdm for a progress bar
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        for x, y, attention_mask, loss_mask in pbar:
            x, y, attention_mask, loss_mask = x.to(device), y.to(device), attention_mask.to(device), loss_mask.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=train_cfg.get("use_amp", False)):
                logits = model(x, attention_mask=attention_mask)

                logits_flat = logits.view(-1, logits.size(-1))
                y_flat = y.view(-1)
                loss_mask_flat = loss_mask.view(-1)

                masked_logits = logits_flat[loss_mask_flat]
                masked_y = y_flat[loss_mask_flat]

                if masked_y.numel() > 0:
                    loss = F.cross_entropy(
                        masked_logits,
                        masked_y,
                        ignore_index=pad_token_id
                    )
                else:
                    # If no tokens are masked for loss, create a zero loss to avoid issues
                    loss = torch.sum(model.parameters().__next__()) * 0.0

            # Backpropagation
            scaler.scale(loss).backward()
            
            # Gradient Clipping
            if train_cfg.get("max_grad_norm") is not None:
                scaler.unscale_(optimizer) # Unscale gradients before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["max_grad_norm"])

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            # Update learning rate scheduler
            lr_scheduler.step()

            total_loss += loss.item()
            global_step += 1

            # Update progress bar
            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        avg_loss = total_loss / len(train_dataloader)
        elapsed_time = (datetime.now() - start_time).total_seconds() / 60
        train_loss.append(avg_loss)
        train_acc.append(total_correct_predictions/len(train_dataloader))
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f} (Elapsed: {elapsed_time:.2f} min)")

        if (epoch + 1) % train_cfg["save_interval"] == 0:
            os.makedirs(train_cfg["save_dir"], exist_ok=True)
            # Save checkpoint with epoch in filename
            checkpoint_path = os.path.join(train_cfg["save_dir"], f"model_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        if (epoch + 1) % train_cfg["eval_interval"] == 0:
            model.eval()
            val_total_loss = 0
            val_total_correct_predictions = 0
            val_total_masked_tokens = 0

            with torch.no_grad():
                for x, y, attention_mask, loss_mask in val_dataloader:
                    x, y, attention_mask, loss_mask = x.to(device), y.to(device), attention_mask.to(device), loss_mask.to(device)
                    
                    with torch.cuda.amp.autocast(enabled=train_cfg.get("use_amp", False)):
                        logits = model(x, attention_mask=attention_mask)

                        logits_flat = logits.view(-1, logits.size(-1))
                        y_flat = y.view(-1)
                        loss_mask_flat = loss_mask.view(-1)

                        masked_logits = logits_flat[loss_mask_flat]
                        masked_y = y_flat[loss_mask_flat]

                        if masked_y.numel() > 0:
                            val_loss = F.cross_entropy(
                                masked_logits,
                                masked_y,
                                ignore_index=pad_token_id
                            )
                            val_total_loss += val_loss.item()

                            # Calculate accuracy
                            predictions = torch.argmax(masked_logits, dim=-1)
                            val_total_correct_predictions += (predictions == masked_y).sum().item()
                            val_total_masked_tokens += masked_y.numel()
                
            avg_val_loss = val_total_loss / len(val_dataloader)
            val_accuracy = val_total_correct_predictions / val_total_masked_tokens if val_total_masked_tokens > 0 else 0.0
            test_loss.append(avg_val_loss)
            test_acc.append(val_accuracy)
            print(f"Validation Loss after Epoch {epoch + 1}: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            model.train()

    print("Training finished.")
    # Save a final checkpoint
    os.makedirs(train_cfg["save_dir"], exist_ok=True)
    final_checkpoint_path = os.path.join(train_cfg["save_dir"], "model_final.pt")
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Final model saved to {final_checkpoint_path}")
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(test_loss, label="Testing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(train_cfg["save_dir"], "loss_plot.png"))
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(test_acc, label="Testing Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(train_cfg["save_dir"], "acc_plot.png"))
    plt.show()


if __name__ == "__main__":
    train()
