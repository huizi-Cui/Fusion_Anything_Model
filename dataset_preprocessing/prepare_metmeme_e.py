"""MET-Meme(E) images and text encoded with ResNet-18 and BERT."""

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
from .prepare_memotion import torch_random_split_indices

DATA_ROOT = dataset_path('datasets/10_MET-Meme-E/data/metmeme')
IMAGE_DIR = dataset_path('datasets/10_MET-Meme-E/data/metmeme/Eimages/Eimages/Eimages')
OUTPUT_DIR = ROOT / "data/metmeme_e_zero"
SEED = 42
CLASS_NAMES = ["happiness", "love", "anger", "sorrow", "fear", "hate", "surprise"]


def load_data(seed: int = SEED):
    text = pd.read_csv(DATA_ROOT/"E_text.csv", encoding="latin1")
    text.dropna(axis=1, how="all", inplace=True)
    try:
        label = pd.read_csv(DATA_ROOT/"label_E.csv", encoding="utf-8")
    except UnicodeDecodeError:
        label = pd.read_csv(DATA_ROOT/"label_E.csv", encoding="ISO-8859-1")
    label.dropna(axis=1, how="all", inplace=True)
    frame = pd.merge(text, label, on="file_name", how="inner").dropna()
    labels = frame["sentiment category"].apply(lambda value: int(str(value).split("(")[0])-1).to_numpy(dtype=np.int64)
    image_paths = [str(IMAGE_DIR/str(name)) for name in frame["file_name"]]
    texts = [clean_text(value) for value in frame["text"].fillna("").astype(str)]
    n=len(frame); n_train=int(0.7*n); n_validation=int(0.1*n); n_test=n-n_train-n_validation
    train, validation, test = torch_random_split_indices(n,[n_train,n_validation,n_test],seed)
    context=np.concatenate([train,validation])
    return image_paths,texts,labels,train,validation,context,test


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT_DIR)
    parser.add_argument("--device",default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",type=int,default=SEED)
    parser.add_argument("--pca48",action="store_true")
    args=parser.parse_args()
    if args.seed != SEED: parser.error("MET-Meme(E) uses the fixed seed-42 split")
    paths,texts,labels,train,validation,context,test=load_data(args.seed)
    output=encode_save_without_normalization(
        output_dir=args.output_dir.resolve(),device=torch.device(args.device),image_paths=paths,texts=texts,
        labels=labels,context_indices=context,test_indices=test,
        split_indices={"train":train,"validation":validation,"context":context,"test":test},
        image_encoder="resnet18",text_encoder="bert",
        info={"dataset":"metmeme_e_zero","display_name":"MET-Meme(E)","source_dataset":"MET-Meme(E)",
              "n_classes":7,"class_names":CLASS_NAMES,"n_train":len(train),"n_validation":len(validation),
              "split":"torch-style fixed random split train/validation/test = 7:1:2, seed 42",
              "context_splits":["train","validation"],"query_split":"test","seed":SEED,
              "target_column":"sentiment category","label_map":"1-based category prefix -> 0-based class",
              "data_dir":str(DATA_ROOT),"image_dir":str(IMAGE_DIR),"text_columns":["text"]},
    )
    if args.pca48: print(project_dataset(output.name,source_root=output.parent,output_root=output.parent))


if __name__ == "__main__": main()

