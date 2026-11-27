"""SIIM-ISIC images encoded with ResNet-18 alongside structured metadata."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from .dataset_paths import dataset_path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .encoders_zero import build_resnet18_encoder, split_without_normalization


DATA_DIR = dataset_path('datasets/18_SIIM-ISIC/data/SIIM-ISIC')
CSV_PATH = dataset_path('datasets/18_SIIM-ISIC/data/SIIM-ISIC/train.csv')
IMAGE_DIR = dataset_path('datasets/18_SIIM-ISIC/data/SIIM-ISIC/jpeg/train')
OUTPUT_DIR = ROOT / "data/siim_zero"
SEED = 42
STRUCTURED_COLUMNS = ["sex", "age_approx", "anatom_site_general_challenge"]


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image)


def multibench_split_indices(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validation_size = int(0.1 * size)
    test_size = int(0.1 * size)
    train_size = size - validation_size - test_size
    subsets = random_split(
        range(size),
        [train_size, validation_size, test_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    return tuple(np.asarray(subset.indices, dtype=np.int64) for subset in subsets)


def extract_image_features(paths: list[Path], device: torch.device) -> np.ndarray:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} JPEG images; first: {missing[0]}")

    encoder = build_resnet18_encoder(device)
    loader = DataLoader(
        ImageDataset(paths), batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True, prefetch_factor=2,
    )
    features = []
    with torch.no_grad():
        for batch in loader:
            features.append(encoder(batch.to(device)).flatten(1).cpu().numpy())
    return np.concatenate(features).astype(np.float32)


def encode_structured(frame: pd.DataFrame, context_size: int) -> np.ndarray:
    frame = frame[STRUCTURED_COLUMNS].copy()
    frame["age_approx"] = pd.to_numeric(frame["age_approx"], errors="coerce")
    age_median = frame.iloc[:context_size]["age_approx"].median()
    frame["age_approx"] = frame["age_approx"].fillna(age_median)

    categorical = ["sex", "anatom_site_general_challenge"]
    frame[categorical] = frame[categorical].fillna("unknown")
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(frame.iloc[:context_size][categorical].astype(str))
    encoded = encoder.transform(frame[categorical].astype(str)).astype(np.float32)
    age = frame[["age_approx"]].to_numpy(dtype=np.float32)
    return np.concatenate([encoded[:, :1], age, encoded[:, 1:]], axis=1)


def main() -> None:
    metadata = pd.read_csv(CSV_PATH)
    labels = metadata["target"].to_numpy(dtype=np.int64)
    train_indices, validation_indices, test_indices = multibench_split_indices(len(metadata))
    context_indices = np.concatenate([train_indices, validation_indices])
    ordered_indices = np.concatenate([context_indices, test_indices])
    context_size = len(context_indices)

    ordered = metadata.iloc[ordered_indices].reset_index(drop=True)
    ordered_labels = labels[ordered_indices]
    image_paths = [IMAGE_DIR / f"{name}.jpg" for name in ordered["image_name"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_features = extract_image_features(image_paths, device)
    structured_features = encode_structured(ordered, context_size)
    context_relative = np.arange(context_size)
    test_relative = np.arange(context_size, len(ordered))

    context_image, test_image = split_without_normalization(
        image_features, context_relative, test_relative,
    )
    context_tabular, test_tabular = split_without_normalization(
        structured_features, context_relative, test_relative,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "context_image.npy", context_image)
    np.save(OUTPUT_DIR / "test_image.npy", test_image)
    np.save(OUTPUT_DIR / "context_tabular.npy", context_tabular)
    np.save(OUTPUT_DIR / "test_tabular.npy", test_tabular)
    np.save(OUTPUT_DIR / "context_labels.npy", ordered_labels[:context_size])
    np.save(OUTPUT_DIR / "test_labels.npy", ordered_labels[context_size:])
    np.savez(
        OUTPUT_DIR / "split_indices.npz",
        train=train_indices, validation=validation_indices,
        context=context_indices, test=test_indices,
    )

    info = {
        "dataset": "siim_zero",
        "n_classes": 2,
        "class_names": ["benign", "malignant"],
        "n_context": context_size,
        "n_test": len(test_indices),
        "official_split_sizes": {
            "train": len(train_indices),
            "validation": len(validation_indices),
            "test": len(test_indices),
        },
        "split": "MultiBench torch.random_split 80/10/10, seed=42; train+validation -> context",
        "modalities": ["image", "tabular"],
        "modality_dims": {"image": 512, "tabular": 3},
        "encoder": "ImageNet ResNet-18 + tabular metadata",
        "structured_columns": STRUCTURED_COLUMNS,
        "normalization": "none; raw encoder outputs retained without dataset-level normalization",
        "normalization_fit_scope": "none",
    }
    with (OUTPUT_DIR / "info.pkl").open("wb") as file:
        pickle.dump(info, file)

    print(f"Saved: {OUTPUT_DIR}")
    print(f"Context={context_size}, test={len(test_indices)}")
    print(f"Context labels={np.bincount(ordered_labels[:context_size])}")
    print(f"Test labels={np.bincount(ordered_labels[context_size:])}")


if __name__ == "__main__":
    main()

