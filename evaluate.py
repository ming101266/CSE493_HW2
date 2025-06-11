import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import yaml
import os
import argparse
from tqdm import tqdm

from model import GPT, GPTConfig
from utils import CharTokenizer, TextDataset, custom_collate_fn
from part2_train import prepare_dataset_tokens

# -------- Evaluation Function --------
def evaluate(config_path="part2config.yaml"):
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    train_config = cfg_dict["train"]

    parser = argparse.ArgumentParser(description="Evaluate a trained GPT model on the test set.")
    parser.add_argument("--config", type=str, default="part2config.yaml",
                        help="Path to the configuration file (config.yaml).")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(train_config["save_dir"], "model_final.pt"),
                        help="Path to the trained model checkpoint (.pt file).")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    model_cfg = cfg_dict["model"]
    train_cfg = cfg_dict["train"]
    data_cfg = cfg_dict["data"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize character tokenizer (by reading all data files to build its vocabulary)
    all_text = []
    data_dir_path = data_cfg.get("data_dir", "data_p97")
    for split_file in ["train.txt", "val.txt", "test.txt"]:
        file_path = os.path.join(data_dir_path, split_file)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                all_text.extend(f.readlines())
    
    if not all_text:
        raise RuntimeError(f"No text found in {data_dir_path}. Cannot initialize tokenizer.")

    tokenizer = CharTokenizer(text_corpus=all_text)
    pad_token_id = tokenizer.pad_token
    print(f"Vocab Size: {tokenizer.vocab_size}")
    print(f"Pad Token ID: {pad_token_id}")

    # Prepare test dataset
    test_expressions = prepare_dataset_tokens(os.path.join(data_dir_path, "test.txt"), tokenizer)
    test_dataset = TextDataset(test_expressions)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=train_cfg["batch_size"], # Use same batch size as training/validation
        shuffle=False, # No need to shuffle for evaluation
        collate_fn=lambda batch: custom_collate_fn(batch, pad_token_id, model_cfg["block_size"], tokenizer),
        num_workers=4
    )
    print(f"Test dataset size: {len(test_dataset)}")

    # Load model
    config = GPTConfig(**model_cfg)
    config.vocab_size = tokenizer.vocab_size # Ensure vocab size matches tokenizer
    model = GPT(config).to(device)

    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Model loaded successfully from {args.checkpoint}")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.checkpoint}. Please ensure training has completed and the path is correct.")
        return
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        return

    # Set model to evaluation mode
    model.eval()

    total_test_loss = 0
    total_correct_predictions = 0
    total_masked_tokens = 0

    # Automatic Mixed Precision (AMP) for evaluation
    use_amp = train_cfg.get("use_amp", False)

    print("\nStarting evaluation on test set...")
    with torch.no_grad():
        # Use tqdm for a progress bar
        pbar = tqdm(test_dataloader, desc="Evaluating Test Set")
        for x, y, attention_mask, loss_mask in pbar:
            x, y, attention_mask, loss_mask = x.to(device), y.to(device), attention_mask.to(device), loss_mask.to(device)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x, attention_mask=attention_mask)

                logits_flat = logits.view(-1, logits.size(-1))
                y_flat = y.view(-1)
                loss_mask_flat = loss_mask.view(-1)

                masked_logits = logits_flat[loss_mask_flat]
                masked_y = y_flat[loss_mask_flat]

                if masked_y.numel() > 0:
                    test_loss = F.cross_entropy(
                        masked_logits,
                        masked_y,
                        ignore_index=pad_token_id
                    )
                    total_test_loss += test_loss.item()

                    # Calculate accuracy
                    predictions = torch.argmax(masked_logits, dim=-1)
                    total_correct_predictions += (predictions == masked_y).sum().item()
                    total_masked_tokens += masked_y.numel()

            pbar.set_postfix(loss=test_loss.item())

    avg_test_loss = total_test_loss / len(test_dataloader)
    test_accuracy = total_correct_predictions / total_masked_tokens if total_masked_tokens > 0 else 0.0

    print(f"\n--- Test Set Results ---")
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Total Correct Predictions: {total_correct_predictions}")
    print(f"Total Masked Tokens: {total_masked_tokens}")
    print(f"------------------------")

if __name__ == "__main__":
    evaluate()