"""SUN RGB-D RGB and depth images encoded with ImageNet ResNet-18."""

from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path
from .dataset_paths import dataset_path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .encoders_zero import build_resnet18_encoder, split_without_normalization


DATA_DIR = dataset_path('datasets/22_SUN-RGBD/data/sun_rgbd')
OUTPUT_DIR = ROOT / "data/sun_rgbd_zero"
IMAGE_SIZE = 224
BATCH_SIZE = 64
EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Merge all 19 original scene categories into 8 semantic categories.  No sample
# is discarded, and the mapping is identical for the fixed train/test folders.
CLASS_NAMES = [
    "bathroom",
    "residential",
    "education",
    "workspace",
    "meeting",
    "dining",
    "service",
    "furniture",
]
CLASS_MERGE = {
    "bathroom": 0,
    "bedroom": 1,
    "living_room": 1,
    "classroom": 2,
    "library": 2,
    "lab": 2,
    "office": 3,
    "home_office": 3,
    "computer_room": 3,
    "study_space": 3,
    "conference_room": 4,
    "discussion_area": 4,
    "lecture_theatre": 4,
    "dining_area": 5,
    "dining_room": 5,
    "kitchen": 6,
    "rest_space": 6,
    "corridor": 6,
    "furniture_store": 7,
}


def load_split(split_dir: Path) -> tuple[list[Path], np.ndarray, Counter]:
    folders = sorted(path for path in split_dir.iterdir() if path.is_dir())
    found = {path.name for path in folders}
    expected = set(CLASS_MERGE)
    if found != expected:
        raise ValueError(
            f"Unexpected classes in {split_dir}: "
            f"missing={sorted(expected - found)}, extra={sorted(found - expected)}"
        )

    paths: list[Path] = []
    labels: list[int] = []
    original_counts: Counter = Counter()
    for folder in folders:
        images = sorted(
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        )
        if not images:
            raise ValueError(f"No images found in {folder}")
        paths.extend(images)
        labels.extend([CLASS_MERGE[folder.name]] * len(images))
        original_counts[folder.name] = len(images)
    return paths, np.asarray(labels, dtype=np.int64), original_counts


def split_rgb_depth(path: Path) -> tuple[Image.Image, Image.Image]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        if width < 2 or width % 2:
            raise ValueError(f"Expected an even-width RGB-depth pair, got {image.size}: {path}")
        middle = width // 2
        return image.crop((0, 0, middle, height)), image.crop((middle, 0, width, height))


def extract_modality(
    encoder: torch.nn.Module,
    paths: list[Path],
    device: torch.device,
    *,
    depth: bool,
) -> np.ndarray:
    from torchvision.transforms import Compose, Normalize, Resize, ToTensor

    transform = Compose([
        Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    features = []
    with torch.inference_mode():
        for start in range(0, len(paths), BATCH_SIZE):
            batch = []
            for path in paths[start:start + BATCH_SIZE]:
                rgb_image, depth_image = split_rgb_depth(path)
                batch.append(transform(depth_image if depth else rgb_image))
            output = encoder(torch.stack(batch).to(device))
            features.append(output.flatten(1).cpu().numpy())
    return np.concatenate(features).astype(np.float32, copy=False)


def class_counts(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    return {name: int(count) for name, count in zip(CLASS_NAMES, counts)}


def main() -> None:
    train_paths, train_labels, train_original_counts = load_split(DATA_DIR / "train")
    test_paths, test_labels, test_original_counts = load_split(DATA_DIR / "test")
    all_paths = train_paths + test_paths
    context_indices = np.arange(len(train_paths))
    test_indices = np.arange(len(train_paths), len(all_paths))

    print(f"SUN RGB-D: {len(CLASS_MERGE)} original classes -> {len(CLASS_NAMES)} classes")
    print(f"  MultiBench fixed split: context=train={len(train_paths)}, test={len(test_paths)}")
    print(f"  Context class counts: {class_counts(train_labels)}")
    print(f"  Test class counts: {class_counts(test_labels)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = build_resnet18_encoder(device)
    print("[RGB] ImageNet ResNet-18")
    rgb_raw = extract_modality(encoder, all_paths, device, depth=False)
    print("[Depth] ImageNet ResNet-18")
    depth_raw = extract_modality(encoder, all_paths, device, depth=True)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    context_rgb, test_rgb = split_without_normalization(
        rgb_raw, context_indices, test_indices
    )
    context_depth, test_depth = split_without_normalization(
        depth_raw, context_indices, test_indices
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "context_rgb.npy", context_rgb.astype(np.float32))
    np.save(OUTPUT_DIR / "test_rgb.npy", test_rgb.astype(np.float32))
    np.save(OUTPUT_DIR / "context_depth.npy", context_depth.astype(np.float32))
    np.save(OUTPUT_DIR / "test_depth.npy", test_depth.astype(np.float32))
    np.save(OUTPUT_DIR / "context_labels.npy", train_labels)
    np.save(OUTPUT_DIR / "test_labels.npy", test_labels)
    np.save(OUTPUT_DIR / "context_indices.npy", np.arange(len(train_paths), dtype=np.int64))
    np.savez(
        OUTPUT_DIR / "split_indices.npz",
        context=context_indices,
        test=test_indices,
    )

    info = {
        "dataset": "sun_rgbd_zero",
        "n_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "n_context": len(train_paths),
        "n_test": len(test_paths),
        "modalities": ["rgb", "depth"],
        "modality_dims": {"rgb": 512, "depth": 512},
        "encoder": "rgb=ImageNet ResNet-18, depth=ImageNet ResNet-18",
        "split_seed": None,
        "split": "MultiBench fixed directories: context=train, test=test; its val=test is not duplicated",
        "original_to_merged": CLASS_MERGE,
        "train_original_class_counts": dict(train_original_counts),
        "test_original_class_counts": dict(test_original_counts),
        "context_class_counts": class_counts(train_labels),
        "test_class_counts": class_counts(test_labels),
        "normalization": "none; raw encoder outputs retained without dataset-level normalization",
        "normalization_fit_scope": "none",
    }
    with (OUTPUT_DIR / "info.pkl").open("wb") as file:
        pickle.dump(info, file)
    print(f"Done: RGB={context_rgb.shape}/{test_rgb.shape}, "
          f"Depth={context_depth.shape}/{test_depth.shape}")
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

