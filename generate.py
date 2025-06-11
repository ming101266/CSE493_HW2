import random
import os
import yaml

def modinv(b, p):
    # Modular inverse using Fermat's little theorem (p prime)
    if b == 0:
        return None
    return pow(b, p-2, p)

def generate_samples(p, n_samples, op):
    samples = []
    while len(samples) < n_samples:
        a = random.randint(0, p)
        b = random.randint(0, p)
        if op == '+':
            c = (a + b) % p
            expr = f"{a}+{b}={c}"
        elif op == '-':
            c = (a - b) % p
            expr = f"{a}-{b}={c}"
        else:  # division
            inv_b = modinv(b, p)
            if inv_b is None:
                continue  # skip division by zero
            c = (a * inv_b) % p
            expr = f"{a}/{b}={c}"
        samples.append(expr)
    return samples

def split_and_save(samples, outdir):
    random.shuffle(samples)
    n = len(samples)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    os.makedirs(outdir, exist_ok=True)

    splits = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }

    for split, data in splits.items():
        path = os.path.join(outdir, f"{split}.txt")
        with open(path, "w") as f:
            for line in data:
                f.write(line + "\n")
        print(f"Saved {len(data)} samples to {path}")

def main(config_path="part2config.yaml"):
    # 0. Load config
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    generate_config = cfg_dict["generate"]
    seed = generate_config["seed"]
    random.seed(seed)
    samples_per_p = generate_config["num_samples"]

    samples = generate_samples(generate_config["prime"], samples_per_p, op = generate_config["op"])
    outdir = f"data"
    split_and_save(samples, outdir)

if __name__ == "__main__":
    main()
