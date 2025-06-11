import random
import os

def modinv(b, p):
    # Modular inverse using Fermat's little theorem (p prime)
    if b == 0:
        return None
    return pow(b, p-2, p)

def generate_samples(p, n_samples):
    ops = ['+', '-']
    samples = []
    while len(samples) < n_samples:
        a = random.randint(0, p)
        b = random.randint(0, p)
        op = random.choice(ops)
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

def main():
    random.seed(42)
    samples_per_p = 20000

    for p in [97, 113]:
        print(f"\nGenerating samples for p={p}...")
        samples = generate_samples(p, samples_per_p)
        outdir = f"data_p{p}"
        split_and_save(samples, outdir)

if __name__ == "__main__":
    main()
