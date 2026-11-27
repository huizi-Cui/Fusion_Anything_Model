"""CMU-MOSI aligned features pooled over valid text steps."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from .dataset_paths import dataset_path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from .encoders_zero import split_without_normalization

DEFAULT_MOSI_PATH = dataset_path('datasets/04_CMU-MOSI/data/CMU-MOSI/mosi_raw.pkl')
DEFAULT_OUTPUT_DIR = ROOT / "data/cmu_mosi_zero"


def masked_mean(features: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    features = np.nan_to_num(np.asarray(features, dtype=np.float32),
                             nan=0.0, posinf=0.0, neginf=0.0)
    weights = valid_mask.astype(np.float32)[..., None]
    counts = weights.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("Encountered a sequence without valid text-aligned steps")
    return ((features * weights).sum(axis=1) / counts).astype(np.float32)


def load_split(split_data: dict):
    vision = np.asarray(split_data["vision"], dtype=np.float32)
    audio = np.asarray(split_data["audio"], dtype=np.float32)
    text = np.asarray(split_data["text"], dtype=np.float32)
    if vision.shape[:2] != audio.shape[:2] or vision.shape[:2] != text.shape[:2]:
        raise ValueError(f"Modalities are not word aligned: {vision.shape}, {audio.shape}, {text.shape}")

    # MultiBench aligned loader starts every modality at the first nonzero text step.
    valid_mask = np.any(text != 0, axis=-1)
    modalities = {
        "vision": masked_mean(vision, valid_mask),
        "audio": masked_mean(audio, valid_mask),
        "text": masked_mean(text, valid_mask),
    }
    scores = np.asarray(split_data["labels"], dtype=np.float32).reshape(-1)
    # Exact MultiBench Affectdataset classification rule: positive iff score > 0.
    labels = (scores > 0).astype(np.int64)
    ids = [str(value) for value in split_data.get("id", range(len(labels)))]
    return modalities, labels, scores, ids, valid_mask.sum(axis=1)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare MultiBench-compatible CMU-MOSI for LimiX")
    parser.add_argument("--mosi_path", type=Path, default=DEFAULT_MOSI_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    with args.mosi_path.open("rb") as handle:
        data = pickle.load(handle)
    if list(data) != ["train", "valid", "test"]:
        raise ValueError(f"Expected MultiBench train/valid/test splits, got {list(data)}")

    loaded = {name: load_split(data[name]) for name in ("train", "valid", "test")}
    context_modalities = {
        modal: np.concatenate((loaded["train"][0][modal], loaded["valid"][0][modal]))
        for modal in ("vision", "audio", "text")
    }
    test_modalities = loaded["test"][0]
    context_labels = np.concatenate((loaded["train"][1], loaded["valid"][1]))
    test_labels = loaded["test"][1]
    context_scores = np.concatenate((loaded["train"][2], loaded["valid"][2]))
    test_scores = loaded["test"][2]
    context_ids = loaded["train"][3] + loaded["valid"][3]
    test_ids = loaded["test"][3]

    # Preserve the temporally pooled MultiBench features exactly as extracted.
    # Dataset-level z-score normalization is intentionally disabled.
    for modal in context_modalities:
        combined = np.concatenate((context_modalities[modal], test_modalities[modal]))
        n_context = len(context_labels)
        context_modalities[modal], test_modalities[modal] = split_without_normalization(
            combined, np.arange(n_context), np.arange(n_context, len(combined))
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for modal in context_modalities:
        np.save(args.output_dir / f"context_{modal}.npy", context_modalities[modal])
        np.save(args.output_dir / f"test_{modal}.npy", test_modalities[modal])
    np.save(args.output_dir / "context_labels.npy", context_labels)
    np.save(args.output_dir / "test_labels.npy", test_labels)
    np.save(args.output_dir / "context_indices.npy", np.arange(len(context_labels), dtype=np.int64))
    np.savez(
        args.output_dir / "split_indices.npz",
        context=np.arange(len(context_labels), dtype=np.int64),
        test=np.arange(len(context_labels), len(context_labels) + len(test_labels), dtype=np.int64),
    )
    with (args.output_dir / "info.pkl").open("wb") as handle:
        pickle.dump({
            "dataset": "mosi_multibench_zero", "n_classes": 2,
            "class_names": ["non_positive", "positive"],
            "n_context": len(context_labels), "n_test": len(test_labels),
            "official_split_sizes": {name: len(loaded[name][1]) for name in loaded},
            "split_policy": "official train+valid -> context; official test -> test",
            "label_rule": "MultiBench classification: score > 0 positive, score <= 0 non-positive",
            "features": "word-aligned FACET_4.2 35d / COVAREP 74d / GloVe 840B 300d",
            "limix_adapter": "remove left padding using text mask, then temporal mean",
            "normalization": "none; raw pooled features retained without dataset-level normalization",
            "normalization_fit_scope": "none",
            "modality_dims": {name: int(value.shape[1]) for name, value in context_modalities.items()},
            "context_scores": context_scores.tolist(), "test_scores": test_scores.tolist(),
            "context_ids": context_ids, "test_ids": test_ids,
            "valid_length_stats": {
                name: {"min": int(loaded[name][4].min()),
                       "median": float(np.median(loaded[name][4])),
                       "max": int(loaded[name][4].max())}
                for name in loaded
            },
        }, handle)
    print(f"Saved MultiBench-compatible MOSI to {args.output_dir}")
    print(f"Context={len(context_labels)}, test={len(test_labels)}")
    print(f"Labels: context={np.bincount(context_labels)}, test={np.bincount(test_labels)}")
    print("Dims:", {name: value.shape[1] for name, value in context_modalities.items()})


if __name__ == "__main__":
    main()

