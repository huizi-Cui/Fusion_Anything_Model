"""MultiOFF images and text encoded with DINOv3 ViT-B/16 and Qwen3-Embedding-0.6B."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from .dataset_paths import dataset_path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .encoders_zero import (
    build_dinov3_vitb16_encoder,
    extract_dinov3_features,
    extract_qwen3_embedding_features,
    split_without_normalization,
)
from .prepare_projection import project_dataset
from .image_text_common import clean_text


DATA_ROOT = dataset_path('datasets/09_MultiOFF/data/MultiOFF')
IMAGE_DIR = dataset_path('datasets/09_MultiOFF/data/MultiOFF/Labelled Images')
SPLIT_DIR = dataset_path('datasets/09_MultiOFF/data/MultiOFF/Split Dataset')
OUTPUT_DIR = ROOT / "data/multioff_zero"

TRAIN_FILE = "Training_meme_dataset.csv"
VALIDATION_FILE = "Validation_meme_dataset.csv"
TEST_FILE = "Testing_meme_dataset.csv"
LABEL_MAP = {"Non-offensiv": 0, "offensive": 1}
CLASS_NAMES = ["non_offensive", "offensive"]
SEED = 42


def load_data():
    """Load official partitions without resampling or reordering rows."""
    train = pd.read_csv(SPLIT_DIR / TRAIN_FILE)
    validation = pd.read_csv(SPLIT_DIR / VALIDATION_FILE)
    test = pd.read_csv(SPLIT_DIR / TEST_FILE)
    frame = pd.concat([train, validation, test], ignore_index=True)

    mapped = frame["label"].map(LABEL_MAP)
    if mapped.isna().any():
        unknown = sorted(frame.loc[mapped.isna(), "label"].astype(str).unique())
        raise ValueError(f"Unknown MultiOFF labels: {unknown}")
    labels = mapped.to_numpy(dtype=np.int64)
    image_paths = [str(IMAGE_DIR / str(name)) for name in frame["image_name"]]
    texts = [clean_text(text) for text in frame["sentence"].fillna("").astype(str)]

    n_train = len(train)
    n_validation = len(validation)
    context_indices = np.arange(n_train + n_validation, dtype=np.int64)
    test_indices = np.arange(n_train + n_validation, len(frame), dtype=np.int64)
    return image_paths, texts, labels, context_indices, test_indices, n_train, n_validation


def _distribution(labels: np.ndarray) -> dict[str, int]:
    return {
        CLASS_NAMES[class_id]: int(np.count_nonzero(labels == class_id))
        for class_id in range(len(CLASS_NAMES))
    }


def prepare(output_dir: Path, device: torch.device) -> Path:
    image_paths, texts, labels, context, test, n_train, n_validation = load_data()
    missing = [path for path in image_paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first={missing[0]}")

    print(
        f"MultiOFF: train={n_train}, validation={n_validation}, "
        f"Context={len(context)}, Test={len(test)}",
        flush=True,
    )
    print("Extracting image features: DINOv3 ViT-B/16", flush=True)
    image_encoder = build_dinov3_vitb16_encoder(device)
    image_features = extract_dinov3_features(image_encoder, image_paths, device)
    del image_encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Extracting text features: Qwen3-Embedding-0.6B", flush=True)
    text_features = extract_qwen3_embedding_features(texts, device)

    context_image, test_image = split_without_normalization(image_features, context, test)
    context_text, test_text = split_without_normalization(text_features, context, test)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "context_image.npy", context_image)
    np.save(output_dir / "test_image.npy", test_image)
    np.save(output_dir / "context_text.npy", context_text)
    np.save(output_dir / "test_text.npy", test_text)
    np.save(output_dir / "context_labels.npy", labels[context])
    np.save(output_dir / "test_labels.npy", labels[test])
    np.save(output_dir / "context_indices.npy", context)
    np.savez(
        output_dir / "split_indices.npz",
        train=np.arange(n_train, dtype=np.int64),
        validation=np.arange(n_train, n_train + n_validation, dtype=np.int64),
        context=context,
        test=test,
    )

    info = {
        "dataset": "multioff_zero",
        "display_name": "MultiOFF",
        "source_dataset": "MultiOFF",
        "n_classes": 2,
        "class_names": CLASS_NAMES,
        "n_train": n_train,
        "n_validation": n_validation,
        "n_context": len(context),
        "n_test": len(test),
        "context_label_distribution": _distribution(labels[context]),
        "test_label_distribution": _distribution(labels[test]),
        "modalities": ["image", "text"],
        "modality_dims": {"image": 768, "text": 1024},
        "structured_modalities": [],
        "encoder": "DINOv3 ViT-B/16 (768d) + Qwen3-Embedding-0.6B (1024d)",
        "image_encoder": "dinov3_b",
        "text_encoder": "qwen3_embedding_0_6b",
        "split": "official CSV train/validation/test; train+validation -> Context",
        "context_splits": ["train", "validation"],
        "query_split": "test",
        "seed": SEED,
        "target_column": "label",
        "label_map": LABEL_MAP,
        "data_dir": str(DATA_ROOT),
        "image_dir": str(IMAGE_DIR),
        "text_columns": ["sentence"],
        "normalization": "none; raw encoder outputs retained without dataset-level normalization",
        "normalization_fit_scope": "none",
        "migration_source": "inference_multimodal_1024d/prepare_zero_imgtxt.py",
    }
    with (output_dir / "info.pkl").open("wb") as file:
        pickle.dump(info, file)
    print(f"Saved Raw features: {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--pca48", action="store_true")
    args = parser.parse_args()
    if args.seed != SEED:
        parser.error("MultiOFF uses the preserved seed 42 treatment")

    output = prepare(args.output_dir.resolve(), torch.device(args.device))
    if args.pca48:
        projected = project_dataset(
            output.name,
            pca_dim=48,
            source_root=output.parent,
            output_root=output.parent,
        )
        print(f"Saved PCA48 features: {projected}", flush=True)


if __name__ == "__main__":
    main()

