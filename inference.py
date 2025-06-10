import torch
import argparse
from model import GPT, GPTConfig
import tiktoken


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
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_epoch400.pt", help="Path to model .pt checkpoint")
    parser.add_argument("--prompt", type=str, default="", help="Initial prompt string")
    parser.add_argument("--tokens", type=int, default=10, help="Number of tokens to generate")
    args = parser.parse_args()

    # Load model config and checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GPTConfig(block_size=12)
    model = GPT(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)

    # Load tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    bos_token_id = tokenizer.eot_token
    input_ids = torch.tensor([[bos_token_id] + tokenizer.encode(args.prompt)], dtype=torch.long).to(device)


    # Generate
    output_text = generate(model, input_ids, args.tokens, tokenizer)


    print("\n=== Generated ===\n")
    print(output_text)

if __name__ == "__main__":
    main()
