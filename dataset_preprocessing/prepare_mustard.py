"""MUStARD uses supplied BERT, audio and ResNet-152 features with the official folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import zipfile
from pathlib import Path
from .dataset_paths import dataset_path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .encoders_zero import split_without_normalization


DATA_DIR = dataset_path('datasets/16_MUStARD/data/MUStARD')
OUTPUT_ROOT = ROOT / "data"
N_CLASSES = 2
CLASS_NAMES = ["non_sarcastic", "sarcastic"]


def load_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file, encoding="latin1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_annotations() -> tuple[list[str], np.ndarray]:
    path = DATA_DIR / "data" / "sarcasm_data.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = list(payload)
    labels = np.asarray([int(payload[item]["sarcasm"]) for item in ids], dtype=np.int64)
    if len(ids) != 690 or not np.array_equal(np.unique(labels), np.arange(N_CLASSES)):
        raise ValueError("Unexpected MUStARD annotations")
    return ids, labels


def bert_cls_from_record(record: dict) -> np.ndarray:
    cls_features = record["features"][0]
    layers = cls_features["layers"]
    if len(layers) < 4:
        raise ValueError("Expected four BERT layers for the CLS token")
    return np.mean(
        [np.asarray(layers[index]["values"], dtype=np.float32) for index in range(4)],
        axis=0,
    ).astype(np.float32)


def load_official_text_features(n_samples: int) -> np.ndarray:
    archive_path = DATA_DIR / "hf_data" / "BERT_text_features.zip"
    records = []
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open("bert-output.jsonl") as file:
            for line in file:
                if line.strip():
                    records.append(bert_cls_from_record(json.loads(line)))
    if len(records) != n_samples:
        raise ValueError(f"BERT rows={len(records)}, annotations={n_samples}")
    features = np.stack(records)
    if features.shape != (n_samples, 768):
        raise ValueError(f"Unexpected BERT shape: {features.shape}")
    return features


def load_official_audio_features(ids: list[str]) -> np.ndarray:
    payload = load_pickle(DATA_DIR / "data" / "audio_features.p")
    missing = set(ids) - set(payload)
    if missing:
        raise ValueError(f"Missing official audio features: {sorted(missing)[:10]}")
    # Official baseline get_target_audio_pool averages axis=1 (the ten time bins).
    features = np.stack([
        np.asarray(payload[item], dtype=np.float32).mean(axis=1) for item in ids
    ])
    if features.shape != (len(ids), 283):
        raise ValueError(f"Unexpected audio shape: {features.shape}")
    return features


def load_official_video_features(ids: list[str]) -> np.ndarray:
    path = DATA_DIR / "hf_data" / "features" / "utterances_final" / "resnet_pool5.hdf5"
    features = []
    with h5py.File(path, "r") as file:
        missing = set(ids) - set(file.keys())
        if missing:
            raise ValueError(f"Missing official video features: {sorted(missing)[:10]}")
        # Official baseline get_target_video_pool averages over video frames.
        for item in ids:
            features.append(np.asarray(file[item], dtype=np.float32).mean(axis=0))
    output = np.stack(features)
    if output.shape != (len(ids), 2048):
        raise ValueError(f"Unexpected video shape: {output.shape}")
    return output


def load_official_folds(n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
    path = DATA_DIR / "data" / "split_indices.p"
    raw_folds = load_pickle(path)
    folds = []
    for fold_number, (context, test) in enumerate(raw_folds, start=1):
        context = np.asarray(context, dtype=np.int64)
        test = np.asarray(test, dtype=np.int64)
        if len(context) != 552 or len(test) != 138:
            raise ValueError(f"Unexpected fold {fold_number} sizes")
        if set(context) & set(test) or set(context) | set(test) != set(range(n_samples)):
            raise ValueError(f"Invalid official fold {fold_number}")
        folds.append((context, test))
    if len(folds) != 5:
        raise ValueError(f"Expected five official folds, got {len(folds)}")
    return folds


def save_fold(
    fold_number: int,
    context_indices: np.ndarray,
    test_indices: np.ndarray,
    ids: list[str],
    labels: np.ndarray,
    modalities: dict[str, np.ndarray],
) -> Path:
    output_dir = OUTPUT_ROOT / f"mustard_zero_fold{fold_number}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, features in modalities.items():
        context, test = split_without_normalization(features, context_indices, test_indices)
        np.save(output_dir / f"context_{name}.npy", context.astype(np.float32))
        np.save(output_dir / f"test_{name}.npy", test.astype(np.float32))
    np.save(output_dir / "context_labels.npy", labels[context_indices])
    np.save(output_dir / "test_labels.npy", labels[test_indices])
    np.savez(
        output_dir / "split_indices.npz",
        context=context_indices,
        test=test_indices,
        utterance_ids=np.asarray(ids),
    )

    info = {
        "dataset": f"mustard_zero_fold{fold_number}",
        "source_dataset": "MUStARD",
        "official_fold": fold_number,
        "n_classes": N_CLASSES,
        "class_names": CLASS_NAMES,
        "n_context": len(context_indices),
        "n_test": len(test_indices),
        "modalities": ["text", "audio", "video"],
        "modality_dims": {"text": 768, "audio": 283, "video": 2048},
        "encoder": (
            "text=official BERT-base-uncased CLS mean of layers -1..-4; "
            "audio=official librosa 283d features averaged over time bins; "
            "video=official ImageNet ResNet-152 pool5 averaged over frames"
        ),
        "split_seed": None,
        "split": f"official MUStARD speaker-dependent stratified five-fold CV, fold {fold_number}",
        "context_class_counts": np.bincount(labels[context_indices], minlength=2).tolist(),
        "test_class_counts": np.bincount(labels[test_indices], minlength=2).tolist(),
        "normalization": "none; raw encoder outputs retained without dataset-level normalization",
        "normalization_fit_scope": "none",
        "source_files": {
            "annotations": str(DATA_DIR / "data" / "sarcasm_data.json"),
            "folds": str(DATA_DIR / "data" / "split_indices.p"),
            "text": str(DATA_DIR / "hf_data" / "BERT_text_features.zip"),
            "audio": str(DATA_DIR / "data" / "audio_features.p"),
            "video": str(DATA_DIR / "hf_data/features/utterances_final/resnet_pool5.hdf5"),
        },
        "source_sha256": {
            "annotations": sha256_file(DATA_DIR / "data" / "sarcasm_data.json"),
            "folds": sha256_file(DATA_DIR / "data" / "split_indices.p"),
        },
    }
    with (output_dir / "info.pkl").open("wb") as file:
        pickle.dump(info, file)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold",
        type=int,
        choices=range(1, 6),
        help="generate only one official fold; by default all five folds are generated",
    )
    args = parser.parse_args()

    ids, labels = load_annotations()
    folds = load_official_folds(len(ids))
    print(f"MUStARD: {len(ids)} samples, labels={np.bincount(labels).tolist()}")
    print("Loading official BERT/audio/ResNet-152 features")
    modalities = {
        "text": load_official_text_features(len(ids)),
        "audio": load_official_audio_features(ids),
        "video": load_official_video_features(ids),
    }
    if not all(np.isfinite(features).all() for features in modalities.values()):
        raise ValueError("Official features contain NaN or Inf")

    selected = [args.fold] if args.fold is not None else range(1, 6)
    for fold_number in selected:
        context, test = folds[fold_number - 1]
        output = save_fold(
            fold_number, context, test, ids, labels, modalities
        )
        print(
            f"Fold {fold_number}: context={len(context)}, test={len(test)}, saved={output}"
        )


if __name__ == "__main__":
    main()

