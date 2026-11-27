"""Memotion images and text encoded with DINOv3 ViT-B/16 and Qwen3-Embedding-0.6B."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from .dataset_paths import dataset_path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .prepare_projection import project_dataset
from .image_text_common import clean_text, encode_save_without_normalization


DATA_ROOT = dataset_path('datasets/12_Memotion/data/memotion')
IMAGE_DIR = dataset_path('datasets/12_Memotion/data/memotion/all_images')
OUTPUT_DIR = ROOT / "data/memotion_zero"
SEED = 42
CLASS_NAMES = ["class_0", "class_1"]


def torch_random_split_indices(n: int, lengths: list[int], seed: int) -> list[np.ndarray]:
    permutation = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).numpy()
    result, start = [], 0
    for length in lengths:
        result.append(permutation[start:start + length])
        start += length
    return result


def load_data(seed: int = SEED):
    frame = pd.read_csv(DATA_ROOT / "all_data.tsv", sep="\t")
    labels = frame["label"].to_numpy(dtype=np.int64)
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError(f"Unexpected Memotion labels: {np.unique(labels).tolist()}")
    image_paths = [str(IMAGE_DIR / str(name)) for name in frame["image_name"]]
    texts = [clean_text(value) for value in frame["text"].fillna("").astype(str)]
    n = len(frame)
    validation_size = int(0.1 * n)
    test_size = int(0.1 * n)
    train_size = n - validation_size - test_size
    train, validation, test = torch_random_split_indices(
        n, [train_size, validation_size, test_size], seed
    )
    context = np.concatenate([train, validation])
    return image_paths, texts, labels, train, validation, context, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--pca48", action="store_true")
    args = parser.parse_args()
    if args.seed != SEED:
        parser.error("Memotion uses the fixed seed-42 split")
    paths, texts, labels, train, validation, context, test = load_data(args.seed)
    output = encode_save_without_normalization(
        output_dir=args.output_dir.resolve(), device=torch.device(args.device),
        image_paths=paths, texts=texts, labels=labels,
        context_indices=context, test_indices=test,
        split_indices={"train": train, "validation": validation, "context": context, "test": test},
        info={"dataset": "memotion_zero", "display_name": "Memotion", "source_dataset": "Memotion",
              "n_classes": 2, "class_names": CLASS_NAMES,
              "n_train": len(train), "n_validation": len(validation),
              "split": "torch-style fixed random split train/validation/test = 8:1:1, seed 42",
              "context_splits": ["train", "validation"], "query_split": "test",
              "seed": SEED, "target_column": "label", "label_map": {},
              "data_dir": str(DATA_ROOT), "image_dir": str(IMAGE_DIR), "text_columns": ["text"]},
    )
    if args.pca48:
        print(project_dataset(output.name, source_root=output.parent, output_root=output.parent))


if __name__ == "__main__":
    main()

