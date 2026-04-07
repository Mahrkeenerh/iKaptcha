"""
Generate synthetic captcha dataset for training.
Run: uv run --with Pillow python generate_dataset.py
     uv run --with Pillow python generate_dataset.py --train-count 15000 --val-count 1000
"""

import argparse
import multiprocessing as mp
import os
from pathlib import Path


def generate_batch(args: tuple) -> int:
    """Generate a batch of captchas. Called per worker process."""
    start_idx, count, output_dir = args
    # Import inside worker to avoid pickling issues with font cache
    from generate_captcha import generate_captcha

    for i in range(count):
        img, label = generate_captcha()
        img.save(output_dir / f"{start_idx + i}_{label}.png")
    return count


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic captcha dataset")
    parser.add_argument("--train-count", type=int, default=15_000)
    parser.add_argument("--val-count", type=int, default=1_000)
    parser.add_argument("--output-dir", type=str, default="dataset_synthetic")
    args = parser.parse_args()

    train_dir = Path(args.output_dir) / "train"
    val_dir = Path(args.output_dir) / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    num_workers = os.cpu_count() or 4

    for split_name, split_dir, total in [
        ("train", train_dir, args.train_count),
        ("val", val_dir, args.val_count),
    ]:
        print(f"Generating {total} images into {split_dir}/ using {num_workers} workers...")

        chunk_size = total // num_workers
        remainder = total % num_workers
        tasks = []
        offset = 0
        for w in range(num_workers):
            n = chunk_size + (1 if w < remainder else 0)
            tasks.append((offset, n, split_dir))
            offset += n

        done = 0
        with mp.Pool(num_workers) as pool:
            for count in pool.imap_unordered(generate_batch, tasks):
                done += count
                print(f"  {split_name}: {done}/{total}", end="\r", flush=True)

        print(f"  {split_name}: {done}/{total} -- done")

    print("\nDataset generation complete.")


if __name__ == "__main__":
    main()
