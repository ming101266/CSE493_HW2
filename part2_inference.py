import torch
import argparse
from model import GPT, GPTConfig
import yaml
import os
from utils import CharTokenizer, NumberTokenizer

# -------- Generate response --------
@torch.no_grad()
def generate(model, idx, max_new_tokens, tokenizer, block_size, temperature=1.0, top_k=None):
    model.eval()

@torch.no_grad()
def generate(model, idx, max_new_tokens, tokenizer, block_size, temperature=1.0, top_k=None):
    model.eval()
    for _ in range(max_new_tokens):
        # Pad if needed
        if idx.size(1) < block_size:
            pad_len = block_size - idx.size(1)
            pad_val = tokenizer.pad_token if hasattr(tokenizer, 'pad_token') else tokenizer.eot_token
            pad_tensor = torch.full((idx.size(0), pad_len), pad_val, dtype=torch.long, device=idx.device)
            idx_cond = torch.cat([pad_tensor, idx], dim=1)
        else:
            idx_cond = idx[:, -block_size:]
        print(idx_cond)

        # Forward pass
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
        print(next_token.item())
        if next_token.item() == tokenizer.eot_token:
            break

    # Remove surrounding EOTs if any
    tokens = idx[0].tolist()
    if tokens and tokens[0] == tokenizer.eot_token:
        tokens = tokens[1:]
    if tokens and tokens[-1] == tokenizer.eot_token:
        tokens = tokens[:-1]
    
    return tokenizer.decode(tokens)

# -------- Inference --------
def main(config_path="part2config.yaml"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="part2_checkpoints/model_final.pt", help="Path to model .pt checkpoint")
    parser.add_argument("--prompt", type=str, default="", help="Initial prompt string")
    parser.add_argument("--tokens", type=int, default=1, help="Number of tokens to generate")
    args = parser.parse_args()

    # Load model config and checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    model_cfg = cfg_dict["model"]
    train_cfg = cfg_dict["train"] 

    config = GPTConfig(**model_cfg)

    # Initialize character tokenizer
    tokenizer = NumberTokenizer()
    
    config.vocab_size = tokenizer.vocab_size

    model = GPT(config).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)

    encoded_prompt = tokenizer.encode(args.prompt)

    # Remove any accidental trailing EOT
    if encoded_prompt and encoded_prompt[-1] == tokenizer.eot_token:
        encoded_prompt = encoded_prompt[:-1]

    # Prepend EOT as BOS
    input_ids = torch.tensor([[tokenizer.eot_token] + encoded_prompt], dtype=torch.long).to(device)
    print(input_ids)

    output_text = generate(model, input_ids, args.tokens, tokenizer, model_cfg["block_size"])

    print("\n=== Prompt ===\n")
    print(args.prompt)
    print("\n=== Generated ===\n")
    print(output_text)

if __name__ == "__main__":
    main()
