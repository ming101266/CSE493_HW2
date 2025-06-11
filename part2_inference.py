import torch
import argparse
from model import GPT, GPTConfig
import yaml
import os
from utils import CharTokenizer # Import from utils.py

# -------- Generate response --------
@torch.no_grad()
def generate(model, idx, max_new_tokens, tokenizer, temperature=1.0, top_k=None):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
        if next_token.item() == tokenizer.eot_token:
            break

    tokens = idx[0].tolist()
    if tokens and tokens[0] == tokenizer.eot_token:
        tokens = tokens[1:]
    if tokens and tokens[-1] == tokenizer.eot_token:
        tokens = tokens[:-1]
    
    return tokenizer.decode(tokens)

# -------- Inference --------
def main(config_path="part2config.yaml"):
        # Load model config and checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    model_cfg = cfg_dict["model"]
    train_cfg = cfg_dict["train"] 
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=train_cfg["save_dir"] + "model_final.pt", help="Path to model .pt checkpoint")
    parser.add_argument("--prompt", type=str, default="", help="Initial prompt string")
    parser.add_argument("--tokens", type=int, default=10, help="Number of tokens to generate")
    parser.add_argument("--data_dir", type=str, default=train_cfg["save_dir"], help="Directory containing the training data (for tokenizer vocab building)")
    args = parser.parse_args()


    config = GPTConfig(**model_cfg)

    # Initialize character tokenizer by reading all data files
    all_text = []
    for split_file in ["train.txt", "val.txt", "test.txt"]:
        file_path = os.path.join(args.data_dir, split_file)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                all_text.extend(f.readlines())
    
    
    tokenizer = CharTokenizer(text_corpus=all_text)
    
    config.vocab_size = tokenizer.vocab_size

    model = GPT(config).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)

    input_ids = torch.tensor([[tokenizer.eot_token] + tokenizer.encode(args.prompt)], dtype=torch.long).to(device)

    output_text = generate(model, input_ids, args.tokens, tokenizer)

    print("\n=== Prompt ===\n")
    print(args.prompt)
    print("\n=== Generated ===\n")
    print(output_text)

if __name__ == "__main__":
    main()