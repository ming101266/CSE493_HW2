import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import yaml
import os
import argparse
from tqdm import tqdm
import math

from model import GPT, GPTConfig
from utils import CharTokenizer, TextDataset, custom_collate_fn, NumberTokenizer

class ModuloDivisionDataset(Dataset):
    def __init__(self, tokenizer, modulus=97):
        self.tokenizer = tokenizer
        self.modulus = modulus
        self.expressions = self._generate_expressions()

    def _generate_expressions(self):
        generated_expressions = []
        inverses = {}
        for i in range(1, self.modulus):
            inverses[i] = pow(i, self.modulus - 2, self.modulus)

        for a in range(self.modulus):
            for b in range(1, self.modulus):
                b_inv = inverses[b]
                c = (a * b_inv) % self.modulus
                expression_str = f"{a} / {b} = {c}"
                generated_expressions.append(expression_str)
        print(f"Generated {len(generated_expressions)} unique modulo division expressions.")
        return generated_expressions

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, idx):
        expression = self.expressions[idx]
        full_tokenized_expression = [self.tokenizer.eot_token] + self.tokenizer.encode(expression)
        return full_tokenized_expression

def evaluate_modulo_division(config_path="part2config.yaml"):
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    model_cfg = cfg_dict["model"]
    train_cfg = cfg_dict["train"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = NumberTokenizer()
    pad_token_id = tokenizer.pad_token
    print(f"Vocab Size: {tokenizer.vocab_size}")
    print(f"Pad Token ID: {pad_token_id}")

    config = GPTConfig(**model_cfg)
    config.vocab_size = tokenizer.vocab_size
    model = GPT(config).to(device)

    parser = argparse.ArgumentParser(description="Evaluate a trained GPT model on all modulo 97 division possibilities.")
    parser.add_argument("--config", type=str, default="part2config.yaml",
                        help="Path to the configuration file (config.yaml).")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(train_cfg["save_dir"], "model_final.pt"),
                        help="Path to the trained model checkpoint (.pt file). Default is model_final.pt.")
    args = parser.parse_args()

    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"Model loaded successfully from {args.checkpoint}")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.checkpoint}. Please ensure training has completed and the path is correct.")
        return
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        return

    modulus = 97
    evaluation_dataset = ModuloDivisionDataset(tokenizer=tokenizer, modulus=modulus)
    evaluation_dataloader = DataLoader(
        evaluation_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=lambda batch: custom_collate_fn(batch, pad_token_id, config.block_size, tokenizer),
        num_workers=4
    )
    print(f"Modulo division evaluation dataset size: {len(evaluation_dataset)}")

    model.eval()

    total_correct_sequences = 0
    total_sequences = 0
    total_tokens_predicted = 0
    total_correct_tokens = 0

    use_amp = train_cfg.get("use_amp", False)

    print("\nStarting evaluation on all modulo division possibilities...")
    with torch.no_grad():
        pbar = tqdm(evaluation_dataloader, desc="Evaluating Modulo Division")
        for i, (x, y, attention_mask, loss_mask) in enumerate(pbar):
            x, y, attention_mask, loss_mask = x.to(device), y.to(device), attention_mask.to(device), loss_mask.to(device)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x, attention_mask=attention_mask)

                logits_flat = logits.view(-1, logits.size(-1))
                y_flat = y.view(-1)
                loss_mask_flat = loss_mask.view(-1)

                masked_logits = logits_flat[loss_mask_flat]
                masked_y = y_flat[loss_mask_flat]

                if masked_y.numel() > 0:
                    predictions = torch.argmax(masked_logits, dim=-1)
                    
                    batch_size = x.size(0)
                    seq_len = x.size(1)

                    for b_idx in range(batch_size):
                        total_sequences += 1

                        seq_loss_mask = loss_mask[b_idx].bool()
                        seq_y = y[b_idx][seq_loss_mask]
                        
                        seq_logits = logits[b_idx].view(-1, logits.size(-1))
                        seq_masked_logits = seq_logits[seq_loss_mask]

                        if seq_masked_logits.numel() > 0:
                            seq_predictions = torch.argmax(seq_masked_logits, dim=-1)
                            
                            is_sequence_correct = (seq_predictions == seq_y).all().item()
                            if is_sequence_correct:
                                total_correct_sequences += 1
                            
                            total_correct_tokens += (seq_predictions == seq_y).sum().item()
                            total_tokens_predicted += seq_y.numel()

            pbar.set_postfix(accuracy=f"{total_correct_sequences / total_sequences:.4f}" if total_sequences > 0 else "N/A")

    sequence_accuracy = total_correct_sequences / total_sequences if total_sequences > 0 else 0.0

    print(f"\n--- Modulo Division (mod {modulus}) Evaluation Results ---")
    print(f"Total Expressions Evaluated: {total_sequences}")
    print(f"Total Correct Sequences (a/b=c): {total_correct_sequences}")
    print(f"Accuracy: {sequence_accuracy:.4f}")
    print(f"----------------------------------------------------------")

if __name__ == "__main__":
    evaluate_modulo_division()