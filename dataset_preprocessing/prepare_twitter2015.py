"""Twitter2015 images and text encoded with ResNet-18 and BERT."""

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

DATA_ROOT = dataset_path('datasets/21_Twitter2015/data/Twitter2015/twitter2015/twitter2015')
IMAGE_DIR = dataset_path('datasets/21_Twitter2015/data/Twitter2015/twitter2015/twitter2015_images')
OUTPUT_DIR = ROOT / "data/twitter2015_zero"
SEED = 42
CLASS_NAMES = ["negative", "neutral", "positive"]
FILES = {"train": "train.tsv", "validation": "dev.tsv", "test": "test.tsv"}


def load_data():
    frames = {name: pd.read_csv(DATA_ROOT / filename, sep="\t") for name, filename in FILES.items()}
    frame = pd.concat(frames.values(), ignore_index=True)
    labels = frame["#1 Label"].to_numpy(dtype=np.int64)
    image_paths = []
    for value in frame["#2 ImageID"]:
        name = str(value)
        if "." not in name:
            name += ".jpg"
        image_paths.append(str(IMAGE_DIR / name))
    merged_text = frame["#3 String"].fillna("").astype(str) + " " + frame["#3 String.1"].fillna("").astype(str)
    texts = [clean_text(value) for value in merged_text]
    n_train, n_validation = len(frames["train"]), len(frames["validation"])
    train = np.arange(n_train, dtype=np.int64)
    validation = np.arange(n_train, n_train+n_validation, dtype=np.int64)
    context = np.arange(n_train+n_validation, dtype=np.int64)
    test = np.arange(n_train+n_validation, len(frame), dtype=np.int64)
    return image_paths, texts, labels, train, validation, context, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--pca48", action="store_true")
    args = parser.parse_args()
    if args.seed != SEED:
        parser.error("Twitter2015 uses the fixed seed-42 split")
    paths, texts, labels, train, validation, context, test = load_data()
    output = encode_save_without_normalization(
        output_dir=args.output_dir.resolve(), device=torch.device(args.device), image_paths=paths,
        texts=texts, labels=labels, context_indices=context, test_indices=test,
        split_indices={"train":train,"validation":validation,"context":context,"test":test},
        image_encoder="resnet18", text_encoder="bert",
        info={"dataset":"twitter2015_zero","display_name":"Twitter2015","source_dataset":"Twitter2015",
              "n_classes":3,"class_names":CLASS_NAMES,"n_train":len(train),"n_validation":len(validation),
              "split":"official TSV train+dev/test; train+dev -> Context","context_splits":["train","validation"],
              "query_split":"test","seed":SEED,"target_column":"#1 Label","label_map":{},
              "data_dir":str(DATA_ROOT),"image_dir":str(IMAGE_DIR),"text_columns":["#3 String","#3 String.1"]},
    )
    if args.pca48:
        print(project_dataset(output.name, source_root=output.parent, output_root=output.parent))


if __name__ == "__main__":
    main()

