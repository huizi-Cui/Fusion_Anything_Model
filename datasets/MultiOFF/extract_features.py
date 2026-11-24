"""Prepare the MultiOFF example from official raw data or existing PCA48 features."""
from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import re
import shutil
import sys
import unicodedata

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion_anything.data import load_pca_dataset
from fusion_anything.preprocessing.projection import project_dataset

DATASET_DIR = Path(__file__).resolve().parent
SPLITS = (("Training_meme_dataset.csv", 445),
          ("Validation_meme_dataset.csv", 149),
          ("Testing_meme_dataset.csv", 149))
LABEL_MAP = {"Non-offensiv": 0, "offensive": 1}
CLASS_NAMES = ["non_offensive", "offensive"]
TASK = "multioff_zero_pca"


def clean_text(value: str) -> str:
    """Keep the established control-character and whitespace cleanup."""
    text = str(value) if value is not None else ""
    text = "".join(character for character in text
                   if character in "\n\t" or unicodedata.category(character) != "Cc")
    text = re.sub(r"[^\S\n]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_data(data_dir: Path, *, enforce_counts: bool = True) -> dict:
    """Validate official partitions and preserve their row and image/text alignment.

    Args:
        data_dir: Directory containing Labelled Images and Split Dataset.
        enforce_counts: Require the official 445/149/149 partition sizes.

    Returns:
        Image paths, cleaned text, integer labels, and Context/Query row indices.

    Raises:
        ValueError: Split sizes, CSV columns, labels, or image paths are invalid.
        OSError: A split CSV or referenced image is unavailable.
    """
    data_dir = Path(data_dir).resolve()
    image_dir = (data_dir / "Labelled Images").resolve()
    frames = []
    for filename, expected in SPLITS:
        frame = pd.read_csv(data_dir / "Split Dataset" / filename)
        missing = {"image_name", "sentence", "label"}.difference(frame.columns)
        if missing:
            raise ValueError(f"{filename}: missing columns {sorted(missing)}")
        if frame.empty or (enforce_counts and len(frame) != expected):
            raise ValueError(f"{filename}: expected {expected} rows, received {len(frame)}")
        if frame["image_name"].isna().any():
            raise ValueError(f"{filename}: missing image_name")
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    mapped = frame["label"].map(LABEL_MAP)
    if mapped.isna().any():
        unknown = sorted(frame.loc[mapped.isna(), "label"].astype(str).unique())
        raise ValueError(f"Unsupported MultiOFF labels: {unknown}")
    paths = []
    for name in frame["image_name"].astype(str):
        path = (image_dir / name).resolve()
        if not path.is_relative_to(image_dir):
            raise ValueError(f"Image path escapes Labelled Images: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(str(path))
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate images across official split rows")
    n_context = len(frames[0]) + len(frames[1])
    labels = mapped.to_numpy(dtype=np.int64)
    if set(np.unique(labels[:n_context])) != {0, 1}:
        raise ValueError("Context must contain both MultiOFF classes")
    return {"image_paths": paths,
            "texts": [clean_text(t) for t in frame["sentence"].fillna("").astype(str)],
            "labels": labels, "context": np.arange(n_context, dtype=np.int64),
            "test": np.arange(n_context, len(frame), dtype=np.int64),
            "n_train": len(frames[0]), "n_validation": len(frames[1])}


def validate_prepared(source: Path):
    """Validate two unstructured PCA48 modalities and the official class/split sizes."""
    source = Path(source).resolve()
    data = load_pca_dataset(source.parent, {"name": TASK, "data_dir": str(source),
                                           "modalities": ["image", "text"],
                                           "structured_modalities": []})
    if (data.n_context, data.n_test, data.n_classes) != (594, 149, 2):
        raise ValueError("Expected MultiOFF Context=594, Query=149, classes=2")
    if data.info["structured_modalities"] or any(b["shape"][1] != 48 for b in data.feature_blocks):
        raise ValueError("MultiOFF image/text must both be unstructured PCA48")
    return data


def import_prepared(source: Path, output_root: Path) -> Path:
    """Copy the evaluated PCA48 files without importing an external project."""
    source = Path(source).resolve()
    destination = Path(output_root).resolve() / TASK
    validate_prepared(source)
    if source == destination:
        return destination
    if destination.exists():
        raise FileExistsError(f"Output exists: {destination}; choose another --output-root")
    files = ["info.pkl", "projection_models.pkl", "context_labels.npy", "test_labels.npy"]
    files += [f"{prefix}_{modality}.npy" for prefix in ("context", "test")
              for modality in ("image", "text")]
    if (source / "context_indices.npy").is_file():
        files.append("context_indices.npy")
    destination.mkdir(parents=True)
    for filename in files:
        shutil.copy2(source / filename, destination / filename)
    validate_prepared(destination)
    return destination


def extract_raw(data_dir: Path, output_root: Path, device, repository: Path,
                weights: Path, text_model: str) -> Path:
    """Extract frozen DINOv3/Qwen3 features, then fit Context-only PCA48."""
    import torch
    from fusion_anything.preprocessing.multioff_encoders import (
        build_dinov3_vitb16_encoder, extract_dinov3_features,
        extract_qwen3_embedding_features,
    )
    prepared = load_data(data_dir)
    raw = output_root / "multioff_zero"
    if raw.exists() or (output_root / TASK).exists():
        raise FileExistsError(f"Output exists under {output_root}; choose another --output-root")
    model = build_dinov3_vitb16_encoder(device, repository, weights)
    image = extract_dinov3_features(model, prepared["image_paths"], device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    text = extract_qwen3_embedding_features(prepared["texts"], device, text_model)
    if image.shape != (743, 768) or text.shape != (743, 1024):
        raise ValueError(f"Encoder dimensions differ from MultiOFF treatment: {image.shape}, {text.shape}")
    if not np.isfinite(image).all() or not np.isfinite(text).all():
        raise ValueError("Encoder output contains non-finite values")
    raw.mkdir(parents=True)
    for name, values in (("image", image), ("text", text), ("labels", prepared["labels"])):
        for prefix, indices in (("context", prepared["context"]), ("test", prepared["test"])):
            np.save(raw / f"{prefix}_{name}.npy", values[indices])
    np.save(raw / "context_indices.npy", prepared["context"])
    info = {"dataset": "multioff_zero", "display_name": "MultiOFF", "n_classes": 2,
            "class_names": CLASS_NAMES, "label_map": LABEL_MAP,
            "n_train": 445, "n_validation": 149, "n_context": 594, "n_test": 149,
            "modality_dims": {"image": 768, "text": 1024}, "structured_modalities": [],
            "encoder": "DINOv3 ViT-B/16 + Qwen3-Embedding-0.6B", "seed": 42,
            "split": "official CSV train+validation -> Context; test -> Query",
            "normalization": "no dataset-level normalization; text embeddings use per-row L2",
            "image_model_repository": str(repository.resolve()), "image_model_weights": str(weights.resolve()),
            "text_model": text_model}
    with (raw / "info.pkl").open("wb") as stream:
        pickle.dump(info, stream)
    output = project_dataset("multioff_zero", source_root=output_root, output_root=output_root)
    validate_prepared(output)
    return output


def main(argv=None) -> int:
    """Validate raw data, import prepared features, or extract from raw inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATASET_DIR / "raw")
    parser.add_argument("--output-root", type=Path, default=DATASET_DIR / "processed")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepared-source", type=Path, help="Import an existing evaluated PCA48 directory.")
    modes.add_argument("--check-only", action="store_true", help="Validate raw CSV/image paths without encoding.")
    parser.add_argument("--dinov3-repo", type=Path)
    parser.add_argument("--dinov3-weights", type=Path)
    parser.add_argument("--text-model", help="Local Qwen3-Embedding-0.6B snapshot or its Hugging Face model ID.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    try:
        if args.prepared_source:
            output = import_prepared(args.prepared_source, args.output_root)
        elif args.check_only:
            data = load_data(args.data_dir)
            print(f"MultiOFF raw inputs valid: Context={len(data['context'])}, Query={len(data['test'])}")
            return 0
        else:
            if not args.dinov3_repo or not args.dinov3_weights or not args.text_model:
                parser.error("Raw extraction needs --dinov3-repo, --dinov3-weights and --text-model")
            import torch
            torch.manual_seed(42)
            np.random.seed(42)
            device = torch.device(args.device)
            if device.type == "cuda":
                torch.cuda.set_device(device)
                torch.cuda.manual_seed_all(42)
            output = extract_raw(args.data_dir.resolve(), args.output_root.resolve(), device,
                                 args.dinov3_repo, args.dinov3_weights, args.text_model)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"MultiOFF prepared features: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
