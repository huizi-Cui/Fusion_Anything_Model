"""Enrico screenshots and wireframes encoded with ImageNet ResNet-18."""

from __future__ import annotations

import csv
import pickle
import sys
from collections import Counter
from pathlib import Path
from .dataset_paths import dataset_path

import numpy as np
import torch
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .encoders_zero import (
    build_resnet18_encoder,
    extract_resnet18_features,
    split_without_normalization,
)


DATA_DIR = dataset_path('datasets/05_Enrico/data/Enrico')
OUTPUT_DIR = ROOT / "data/enrico_zero"
SEED = 0
BATCH_SIZE = 64

# The 20 official UI topics are exhaustively merged into 10 functional groups.
# This retains every sample and applies the same mapping to all partitions.
CLASS_GROUPS = {
    "generic": ["bare", "other"],
    "communication": ["calculator", "chat"],
    "media": ["camera", "gallery", "mediaplayer"],
    "input_editing": ["editor", "form"],
    "content_feed": ["list", "news"],
    "account": ["login", "profile"],
    "discovery": ["maps", "search"],
    "navigation_system": ["menu", "settings"],
    "dialog_legal": ["modal", "terms"],
    "tutorial": ["tutorial"],
}
CLASS_NAMES = list(CLASS_GROUPS)


def load_annotations() -> tuple[list[str], list[str]]:
    annotation_path = DATA_DIR / "design_topics.csv"
    with annotation_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows or set(rows[0]) != {"screen_id", "topic"}:
        raise ValueError(f"Unexpected annotation format: {annotation_path}")

    screen_ids = [row["screen_id"].strip() for row in rows]
    topics = [row["topic"].strip().lower() for row in rows]
    if len(screen_ids) != len(set(screen_ids)):
        raise ValueError("design_topics.csv contains duplicate screen IDs")
    return screen_ids, topics


def merge_labels(topics: list[str]) -> np.ndarray:
    topic_to_class = {
        topic: class_id
        for class_id, topic_names in enumerate(CLASS_GROUPS.values())
        for topic in topic_names
    }
    observed = set(topics)
    expected = set(topic_to_class)
    if observed != expected:
        raise ValueError(
            "The semantic merge must cover the official topics exactly: "
            f"missing={sorted(observed - expected)}, extra={sorted(expected - observed)}"
        )
    return np.asarray([topic_to_class[topic] for topic in topics], dtype=np.int64)


def official_protocol_split(topics: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the paper's 80/20 split and 15%-of-train validation ratio."""
    indices = np.arange(len(topics), dtype=np.int64)
    context_indices, test_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=SEED,
    )
    train_indices, validation_indices = train_test_split(
        context_indices,
        test_size=0.15,
        random_state=SEED,
    )
    return (
        np.sort(train_indices),
        np.sort(validation_indices),
        np.sort(test_indices),
    )


def validate_files(screen_ids: list[str]) -> tuple[list[Path], list[Path]]:
    screenshots = [DATA_DIR / "screenshots" / f"{screen_id}.jpg" for screen_id in screen_ids]
    wireframes = [DATA_DIR / "wireframes" / f"{screen_id}.png" for screen_id in screen_ids]
    missing = [path for path in screenshots + wireframes if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} Enrico images; first files: {missing[:10]}")
    return screenshots, wireframes


def count_labels(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    return {name: int(count) for name, count in zip(CLASS_NAMES, counts)}


def main() -> None:
    screen_ids, topics = load_annotations()
    labels = merge_labels(topics)
    screenshots, wireframes = validate_files(screen_ids)
    train_indices, validation_indices, test_indices = official_protocol_split(topics)
    context_indices = np.sort(np.concatenate([train_indices, validation_indices]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = build_resnet18_encoder(device)
    print(f"Enrico: {len(topics)} samples, 20 official topics -> {len(CLASS_NAMES)} classes")
    print(
        f"Official paper protocol: train={len(train_indices)}, "
        f"validation={len(validation_indices)}, test={len(test_indices)}"
    )

    print("[Screenshot] ImageNet ResNet-18")
    screenshot_raw = extract_resnet18_features(
        encoder, screenshots, device, batch_size=BATCH_SIZE, num_workers=0
    ).astype(np.float32, copy=False)
    print("[Wireframe] ImageNet ResNet-18")
    wireframe_raw = extract_resnet18_features(
        encoder, wireframes, device, batch_size=BATCH_SIZE, num_workers=0
    ).astype(np.float32, copy=False)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    context_screenshot, test_screenshot = split_without_normalization(
        screenshot_raw, context_indices, test_indices
    )
    context_wireframe, test_wireframe = split_without_normalization(
        wireframe_raw, context_indices, test_indices
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "context_screenshot.npy", context_screenshot.astype(np.float32))
    np.save(OUTPUT_DIR / "test_screenshot.npy", test_screenshot.astype(np.float32))
    np.save(OUTPUT_DIR / "context_wireframe.npy", context_wireframe.astype(np.float32))
    np.save(OUTPUT_DIR / "test_wireframe.npy", test_wireframe.astype(np.float32))
    np.save(OUTPUT_DIR / "context_labels.npy", labels[context_indices])
    np.save(OUTPUT_DIR / "test_labels.npy", labels[test_indices])
    np.save(OUTPUT_DIR / "context_indices.npy", np.arange(len(context_indices), dtype=np.int64))
    np.savez(
        OUTPUT_DIR / "split_indices.npz",
        train=train_indices,
        validation=validation_indices,
        context=context_indices,
        test=test_indices,
    )

    original_counts = Counter(topics)
    info = {
        "dataset": "enrico_zero",
        "n_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "n_context": len(context_indices),
        "n_test": len(test_indices),
        "modalities": ["screenshot", "wireframe"],
        "modality_dims": {"screenshot": 512, "wireframe": 512},
        "encoder": "ImageNet ResNet-18 for screenshot and semantic wireframe",
        "split_seed": SEED,
        "split": (
            "Official-paper random protocol: 80% development/20% test, with 15% "
            "of development designated validation; random_state=0 makes the "
            "otherwise unspecified draw reproducible. Train+validation -> context. The paper publishes "
            "no fixed sample IDs or random seed."
        ),
        "official_split_sizes": {
            "train": len(train_indices),
            "validation": len(validation_indices),
            "context": len(context_indices),
            "test": len(test_indices),
        },
        "label_protocol": "exhaustive semantic merge of all 20 official topics into 10 classes",
        "class_groups": CLASS_GROUPS,
        "original_topic_counts": dict(sorted(original_counts.items())),
        "context_class_counts": count_labels(labels[context_indices]),
        "test_class_counts": count_labels(labels[test_indices]),
        "normalization_fit_scope": "none",
    }
    with (OUTPUT_DIR / "info.pkl").open("wb") as file:
        pickle.dump(info, file)

    print(f"Context class counts: {info['context_class_counts']}")
    print(f"Test class counts: {info['test_class_counts']}")
    print(
        f"Saved to {OUTPUT_DIR}: screenshot={context_screenshot.shape}/{test_screenshot.shape}, "
        f"wireframe={context_wireframe.shape}/{test_wireframe.shape}"
    )


if __name__ == "__main__":
    main()

