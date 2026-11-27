"""Twitter1517 images and text encoded with DINOv3 ViT-B/16 and Qwen3-Embedding-0.6B."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from .dataset_paths import dataset_path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .prepare_projection import project_dataset
from .image_text_common import clean_text, encode_save_without_normalization


DATA_ROOT = dataset_path('datasets/13_Twitter1517/data/Twitter1517')
IMAGE_DIR = dataset_path('datasets/13_Twitter1517/data/Twitter1517/image')
OUTPUT_DIR = ROOT / "data/twitter1517_zero"
SEED = 42
CLASS_NAMES = ["uncorrelated", "correlated"]


def load_data(seed: int = SEED):
    frame = pd.read_csv(DATA_ROOT / "Twitter1517_texts_and_labels.csv")
    labels = frame["cor_label"].to_numpy(dtype=np.int64)
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError(f"Unexpected Twitter1517 labels: {np.unique(labels).tolist()}")
    image_paths = []
    for value in frame["new_image_id"]:
        name = str(value)
        if "." not in name:
            name = f"{name}.jpg"
        image_paths.append(str(IMAGE_DIR / name))
    texts = [clean_text(value) for value in frame["txt"].fillna("").astype(str)]
    indices = np.arange(len(frame))
    train, remainder = train_test_split(
        indices, test_size=0.3, stratify=labels, random_state=seed
    )
    validation, test = train_test_split(
        remainder, test_size=2 / 3, stratify=labels[remainder], random_state=seed
    )
    context = np.sort(np.concatenate([train, validation]))
    return image_paths, texts, labels, train, validation, context, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--pca48", action="store_true")
    args = parser.parse_args()
    if args.seed != SEED:
        parser.error("Twitter1517 uses the fixed seed-42 split")
    paths, texts, labels, train, validation, context, test = load_data(args.seed)
    output = encode_save_without_normalization(
        output_dir=args.output_dir.resolve(), device=torch.device(args.device),
        image_paths=paths, texts=texts, labels=labels,
        context_indices=context, test_indices=test,
        split_indices={"train": train, "validation": validation, "context": context, "test": test},
        info={"dataset": "twitter1517_zero", "display_name": "Twitter1517", "source_dataset": "Twitter1517",
              "n_classes": 2, "class_names": CLASS_NAMES,
              "n_train": len(train), "n_validation": len(validation),
              "split": "fixed-seed stratified train/validation/test = 7:1:2, seed 42",
              "context_splits": ["train", "validation"], "query_split": "test",
              "seed": SEED, "target_column": "cor_label", "label_map": {},
              "data_dir": str(DATA_ROOT), "image_dir": str(IMAGE_DIR), "text_columns": ["txt"]},
    )
    if args.pca48:
        print(project_dataset(output.name, source_root=output.parent, output_root=output.parent))


if __name__ == "__main__":
    main()

