"""ForestNet RGB and SRTM encoded with ResNet-152, with location and year retained as tabular features."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from .dataset_paths import dataset_path
from typing import Any

import numpy as np
import torch


EXTRACT_ROOT = Path(__file__).resolve().parents[1]
if str(EXTRACT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTRACT_ROOT))

from .migration_common import save_dataset


DEFAULT_FORESTNET_DIR = dataset_path('datasets/15_ForestNet/data/ForestNetDataset')
DEFAULT_OUTPUT_ROOT = EXTRACT_ROOT / "data"
DEFAULT_ENCODED_ROOT = DEFAULT_OUTPUT_ROOT / "_encoded_cache"

SOURCE_DATASET = "forestnet"
SOURCE_OUTPUT_NAME = "forestnet_multibench_resnet152_official_split_encoded"
OUTPUT_NAME = "forestnet_multibench_resnet152_official_split"
EXPECTED_MODALITIES = ("image", "srtm", "tabular")
EXPECTED_STRUCTURED_MODALITIES = ("tabular",)
EXPECTED_COUNTS = {"train": 1616, "validation": 473, "test": 668, "pool": 2757}
EXPECTED_CLASS_NAMES = (
    "Grassland shrubland",
    "Other",
    "Plantation",
    "Smallholder agriculture",
)
EXPECTED_LABEL_MAP = {name: index for index, name in enumerate(EXPECTED_CLASS_NAMES)}
EXPECTED_ENCODER = "torchvision_resnet152_imagenet1k_v1_shared_frozen"
EXPECTED_ENCODER_DIM = 2048


def load_info(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected dict metadata in {path}, got {type(value)!r}")
    return value


def validate_source_csvs(forestnet_dir: Path) -> None:
    """Ensure current source CSV membership matches the frozen split counts."""

    import pandas as pd

    filenames = {"train": "train.csv", "validation": "val.csv", "test": "test.csv"}
    required_columns = {
        "merged_label",
        "latitude",
        "longitude",
        "year",
        "example_path",
    }
    for split_name, filename in filenames.items():
        path = forestnet_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"ForestNet split CSV is missing: {path}")
        frame = pd.read_csv(path)
        if len(frame) != EXPECTED_COUNTS[split_name]:
            raise ValueError(
                f"ForestNet {split_name} count drift: {len(frame)} vs "
                f"{EXPECTED_COUNTS[split_name]}"
            )
        missing = required_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"ForestNet {path} lacks columns: {sorted(missing)}")
        if set(frame["merged_label"]) != set(EXPECTED_CLASS_NAMES):
            raise ValueError(f"ForestNet classes changed in {path}")
        tabular = frame[["latitude", "longitude", "year"]].to_numpy(dtype=np.float64)
        if not np.isfinite(tabular).all():
            raise ValueError(f"ForestNet non-finite tabular values in {path}")


def validate_encoder_contract(info: dict[str, Any]) -> None:
    detail = info.get("encoder_detail", {})
    image = detail.get("image", {})
    srtm = detail.get("srtm", {})
    tabular = detail.get("tabular", {})
    if image.get("name") != EXPECTED_ENCODER or image.get("frozen") is not True:
        raise ValueError(f"ForestNet image encoder drift: {image!r}")
    if image.get("input_transform") != "resize_224_to_tensor_imagenet_normalize":
        raise ValueError(f"ForestNet RGB transform drift: {image!r}")
    if srtm.get("name") != EXPECTED_ENCODER or srtm.get("frozen") is not True:
        raise ValueError(f"ForestNet SRTM encoder drift: {srtm!r}")
    if srtm.get("input_transform") != "raw_float32_three_channel_tensor":
        raise ValueError(f"ForestNet SRTM transform drift: {srtm!r}")
    if image.get("weight_sharing") != "shared_with_srtm" or srtm.get(
        "weight_sharing"
    ) != "shared_with_image":
        raise ValueError("ForestNet RGB/SRTM must share one ResNet-152 instance")
    if tabular.get("name") != "identity" or tabular.get("encoded") is not False:
        raise ValueError(f"ForestNet structured encoder drift: {tabular!r}")
    if "latitude, longitude, year" not in str(tabular.get("source", "")):
        raise ValueError(f"ForestNet structured column drift: {tabular!r}")


def validate_encoded_contract(encoded_dir: Path, info: dict[str, Any]) -> None:
    """Reject caches that differ from the approved ForestNet protocol."""

    if info.get("source_dataset") != SOURCE_DATASET:
        raise ValueError(
            f"Wrong source dataset in {encoded_dir}: {info.get('source_dataset')!r}"
        )
    if tuple(info.get("modalities", ())) != EXPECTED_MODALITIES:
        raise ValueError(f"ForestNet modality-order drift: {info.get('modalities')!r}")
    if tuple(info.get("structured_modalities", ())) != EXPECTED_STRUCTURED_MODALITIES:
        raise ValueError(
            f"ForestNet structured-modality drift: {info.get('structured_modalities')!r}"
        )
    if tuple(info.get("class_names", ())) != EXPECTED_CLASS_NAMES:
        raise ValueError(f"ForestNet class-name drift: {info.get('class_names')!r}")
    if info.get("label_map") != EXPECTED_LABEL_MAP:
        raise ValueError(f"ForestNet label-map drift: {info.get('label_map')!r}")
    if int(info.get("n_classes", -1)) != 4 or info.get("dropped_classes") != {}:
        raise ValueError("ForestNet must retain all four merged-label classes")
    validate_encoder_contract(info)

    split = info.get("split", {})
    policy = str(split.get("policy", ""))
    if "official ForestNet train.csv/val.csv/test.csv" not in policy:
        raise ValueError(f"ForestNet split-policy drift: {policy!r}")
    if split.get("seed") is not None:
        raise ValueError("ForestNet official split must not invent a random seed")
    counts = {str(key): int(value) for key, value in split.get("counts", {}).items()}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"ForestNet split-count drift: {counts!r}")

    dimensions: dict[str, int] = {}
    joined_indices: list[np.ndarray] = []
    expected_classes = np.arange(4, dtype=np.int64)
    for split_name in ("train", "validation", "test"):
        labels_path = encoded_dir / f"{split_name}_labels.npy"
        indices_path = encoded_dir / f"{split_name}_pool_indices.npy"
        if not labels_path.is_file() or not indices_path.is_file():
            raise FileNotFoundError(f"Missing frozen ForestNet {split_name} files")
        labels = np.load(labels_path, mmap_mode="r")
        indices = np.load(indices_path, mmap_mode="r")
        if labels.shape != indices.shape or len(labels) != EXPECTED_COUNTS[split_name]:
            raise ValueError(f"ForestNet {split_name} membership/label mismatch")
        if not np.array_equal(np.unique(labels), expected_classes):
            raise ValueError(f"ForestNet {split_name} lacks one or more classes")
        joined_indices.append(np.asarray(indices, dtype=np.int64))
        for modality in EXPECTED_MODALITIES:
            path = encoded_dir / f"{split_name}_{modality}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"Missing ForestNet features: {path}")
            values = np.load(path, mmap_mode="r")
            if values.ndim != 2 or len(values) != len(labels):
                raise ValueError(f"Invalid ForestNet feature shape: {path} {values.shape}")
            if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
                raise ValueError(f"Invalid ForestNet feature values: {path}")
            previous = dimensions.setdefault(modality, int(values.shape[1]))
            if int(values.shape[1]) != previous:
                raise ValueError(f"ForestNet {modality} dimension differs across splits")
    if dimensions.get("tabular") != 3:
        raise ValueError(f"ForestNet tabular dimension must be 3: {dimensions!r}")
    metadata_dimensions = {
        str(key): int(value) for key, value in info.get("dimensions", {}).items()
    }
    if dimensions != metadata_dimensions:
        raise ValueError(
            f"ForestNet dimension metadata drift: arrays={dimensions}, "
            f"info={metadata_dimensions}"
        )
    joined = np.concatenate(joined_indices)
    if not np.array_equal(joined, np.arange(EXPECTED_COUNTS["pool"], dtype=np.int64)):
        raise ValueError("ForestNet official CSV split order/membership changed")


def materialize_upstream_encoded(args: argparse.Namespace) -> Path:
    """Extract the MultiBench-aligned frozen ResNet-152 cache once."""

    validate_source_csvs(args.forestnet_data_dir)
    encoded_dir = args.encoded_root / SOURCE_OUTPUT_NAME
    incomplete = encoded_dir.with_name(encoded_dir.name + ".incomplete")
    if incomplete.exists():
        raise RuntimeError(
            f"Incomplete upstream extraction exists: {incomplete}. Inspect it explicitly."
        )
    if not encoded_dir.is_dir():
        import pandas as pd
        from PIL import Image
        from torchvision.models import ResNet152_Weights, resnet152
        from torchvision.transforms import Compose, Normalize, Resize, ToTensor

        device = torch.device(args.device)
        weights = ResNet152_Weights.IMAGENET1K_V1
        backbone = resnet152(weights=weights)
        encoder = torch.nn.Sequential(
            *list(backbone.children())[:-2], torch.nn.AdaptiveAvgPool2d(1)
        ).to(device).eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)

        rgb_transform = Compose(
            [
                Resize((224, 224)),
                ToTensor(),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        def encode_rgb(paths: list[Path]) -> np.ndarray:
            outputs: list[np.ndarray] = []
            with torch.inference_mode():
                for start in range(0, len(paths), args.batch_size):
                    batch = torch.stack(
                        [
                            rgb_transform(Image.open(path).convert("RGB"))
                            for path in paths[start : start + args.batch_size]
                        ]
                    ).to(device)
                    outputs.append(encoder(batch).flatten(1).cpu().numpy())
            return np.concatenate(outputs).astype(np.float32, copy=False)

        def encode_srtm(paths: list[Path]) -> np.ndarray:
            outputs: list[np.ndarray] = []
            with torch.inference_mode():
                for start in range(0, len(paths), args.batch_size):
                    batch = torch.stack(
                        [
                            torch.from_numpy(
                                np.asarray(np.load(path), dtype=np.float32)
                            )
                            for path in paths[start : start + args.batch_size]
                        ]
                    ).to(device)
                    if batch.ndim != 4 or batch.shape[1] != 3:
                        raise ValueError(
                            f"ForestNet SRTM must be NCHW with three channels: "
                            f"{tuple(batch.shape)}"
                        )
                    outputs.append(encoder(batch).flatten(1).cpu().numpy())
            return np.concatenate(outputs).astype(np.float32, copy=False)

        args.encoded_root.mkdir(parents=True, exist_ok=True)
        incomplete.mkdir()
        filenames = {
            "train": "train.csv",
            "validation": "val.csv",
            "test": "test.csv",
        }
        pool_offset = 0
        split_counts: dict[str, int] = {}
        for split_name, filename in filenames.items():
            frame = pd.read_csv(args.forestnet_data_dir / filename)
            folders = [args.forestnet_data_dir / value for value in frame["example_path"]]
            image_paths = [folder / "images/visible/composite.png" for folder in folders]
            srtm_paths = [folder / "auxiliary/srtm.npy" for folder in folders]
            missing = [path for path in (*image_paths, *srtm_paths) if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"ForestNet source file is missing: {missing[0]}")
            labels = frame["merged_label"].map(EXPECTED_LABEL_MAP)
            if labels.isna().any():
                raise ValueError(f"ForestNet unknown label in {filename}")
            tabular = frame[["latitude", "longitude", "year"]].to_numpy(
                dtype=np.float32
            )
            np.save(incomplete / f"{split_name}_image.npy", encode_rgb(image_paths))
            np.save(incomplete / f"{split_name}_srtm.npy", encode_srtm(srtm_paths))
            np.save(incomplete / f"{split_name}_tabular.npy", tabular)
            np.save(
                incomplete / f"{split_name}_labels.npy",
                labels.to_numpy(dtype=np.int64),
            )
            np.save(
                incomplete / f"{split_name}_pool_indices.npy",
                np.arange(pool_offset, pool_offset + len(frame), dtype=np.int64),
            )
            split_counts[split_name] = len(frame)
            pool_offset += len(frame)

        info = {
            "dataset": SOURCE_OUTPUT_NAME,
            "source_dataset": SOURCE_DATASET,
            "modalities": list(EXPECTED_MODALITIES),
            "structured_modalities": list(EXPECTED_STRUCTURED_MODALITIES),
            "dimensions": {
                "image": EXPECTED_ENCODER_DIM,
                "srtm": EXPECTED_ENCODER_DIM,
                "tabular": 3,
            },
            "class_names": list(EXPECTED_CLASS_NAMES),
            "label_map": EXPECTED_LABEL_MAP,
            "n_classes": 4,
            "dropped_classes": {},
            "split": {
                "policy": "official ForestNet train.csv/val.csv/test.csv",
                "seed": None,
                "counts": {**split_counts, "pool": pool_offset},
            },
            "encoder": (
                "shared torchvision ResNet-152 ImageNet-1K V1 for visible RGB "
                "and raw three-channel SRTM; identity structured tabular"
            ),
            "encoder_detail": {
                "image": {
                    "name": EXPECTED_ENCODER,
                    "weights": "ResNet152_Weights.IMAGENET1K_V1",
                    "frozen": True,
                    "output_dim": EXPECTED_ENCODER_DIM,
                    "input_transform": "resize_224_to_tensor_imagenet_normalize",
                    "weight_sharing": "shared_with_srtm",
                },
                "srtm": {
                    "name": EXPECTED_ENCODER,
                    "weights": "ResNet152_Weights.IMAGENET1K_V1",
                    "frozen": True,
                    "output_dim": EXPECTED_ENCODER_DIM,
                    "input_transform": "raw_float32_three_channel_tensor",
                    "weight_sharing": "shared_with_image",
                },
                "tabular": {
                    "name": "identity",
                    "encoded": False,
                    "source": "raw latitude, longitude, year",
                    "output_dim": 3,
                },
            },
            "responsibilities": [
                "official CSV membership and order",
                "MultiBench-compatible RGB and SRTM input transforms",
                "shared frozen ResNet-152 feature extraction",
            ],
            "excluded_operations": ["dataset-level normalization", "PCA"],
            "alignment_note": (
                "MultiBench trains this shared ResNet-152 separately for every "
                "fusion run. This cache aligns its ImageNet initialization and input "
                "protocol but freezes it to produce one deterministic feature set."
            ),
        }
        with (incomplete / "info.pkl").open("wb") as handle:
            pickle.dump(info, handle)
        incomplete.rename(encoded_dir)
    info = load_info(encoded_dir / "info.pkl")
    validate_encoded_contract(encoded_dir, info)
    return encoded_dir


def convert_encoded_dataset(
    encoded_dir: Path,
    output_root: Path,
    *,
    pca48: bool,
) -> tuple[Path, Path | None]:
    """Build Context/Test and save raw features and optionally create PCA48."""

    info = load_info(encoded_dir / "info.pkl")
    validate_encoded_contract(encoded_dir, info)
    output_dir = output_root / OUTPUT_NAME
    pca_dir = output_root / f"{OUTPUT_NAME}_pca"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_dir}")
    if pca48 and pca_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing PCA dataset: {pca_dir}")

    split_names = ("train", "validation", "test")
    split_labels = {
        name: np.asarray(np.load(encoded_dir / f"{name}_labels.npy"), dtype=np.int64)
        for name in split_names
    }
    n_train = len(split_labels["train"])
    n_validation = len(split_labels["validation"])
    n_test = len(split_labels["test"])
    n_context = n_train + n_validation
    labels = np.concatenate([split_labels[name] for name in split_names])
    train_indices = np.arange(n_train, dtype=np.int64)
    validation_indices = np.arange(n_train, n_context, dtype=np.int64)
    context_indices = np.arange(n_context, dtype=np.int64)
    test_indices = np.arange(n_context, n_context + n_test, dtype=np.int64)

    features = {
        modality: np.concatenate(
            [
                np.asarray(
                    np.load(encoded_dir / f"{name}_{modality}.npy", mmap_mode="r"),
                    dtype=np.float32,
                )
                for name in split_names
            ],
            axis=0,
        )
        for modality in EXPECTED_MODALITIES
    }
    pool_indices = {
        name: np.asarray(
            np.load(encoded_dir / f"{name}_pool_indices.npy"), dtype=np.int64
        )
        for name in split_names
    }
    metadata = dict(info)
    metadata.update(
        dataset=OUTPUT_NAME,
        display_name="ForestNet",
        n_train=n_train,
        n_validation=n_validation,
        n_context=n_context,
        n_test=n_test,
        context_construction="ordered official train followed by validation",
        context_source_splits=["train", "validation"],
        test_source_split="official test",
        structured_modalities=["tabular"],
        structured_columns=["latitude", "longitude", "year"],
        upstream_encoded_dir=str(encoded_dir.resolve()),
        upstream_responsibilities=info.get("responsibilities", []),
        upstream_excluded_operations=info.get("excluded_operations", []),
        downstream_operations=[
            "train+validation Context construction",
            "no dataset-level normalization on every modality",
            *(
                ["Context-only PCA48 for image/SRTM; retain 3d tabular"]
                if pca48
                else []
            ),
        ],
        split_note=(
            "The source CSV train/validation/test membership and order are kept; "
            "only train and validation are combined into Context."
        ),
    )
    metadata.pop("excluded_operations", None)
    save_dataset(
        output_dir,
        labels,
        context_indices,
        test_indices,
        features,
        metadata,
        {
            "train": train_indices,
            "validation": validation_indices,
            "context": context_indices,
            "test": test_indices,
            "train_pool_indices": pool_indices["train"],
            "validation_pool_indices": pool_indices["validation"],
            "test_pool_indices": pool_indices["test"],
        },
        pca48,
    )
    return output_dir, pca_dir if pca48 else None


def plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset": "ForestNet",
        "source": str(args.forestnet_data_dir.resolve()),
        "upstream_split": "keep CSV train/validation/test=1616/473/668",
        "split_seed": None,
        "class_policy": {
            "class_names": list(EXPECTED_CLASS_NAMES),
            "mapped_labels": "0..3",
            "dropped_classes": [],
        },
        "context": "ordered official train+validation",
        "test": "official test",
        "modalities": {
            "image": "visible RGB encoded with shared frozen ImageNet ResNet-152",
            "srtm": "raw 3-channel SRTM encoded with the same frozen ResNet-152",
            "tabular": "raw latitude/longitude/year; identity; structured",
        },
        "normalization": "none; raw encoder outputs retained without dataset-level normalization",
        "pca48": bool(args.pca48),
        "pca_policy": "image/SRTM Context-only PCA48; keep tabular at 3d",
        "encoded_cache": str((args.encoded_root / SOURCE_OUTPUT_NAME).resolve()),
        "output": str((args.output_root / OUTPUT_NAME).resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--pca48", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--encoded-root", type=Path, default=DEFAULT_ENCODED_ROOT)
    parser.add_argument(
        "--forestnet-data-dir", type=Path, default=DEFAULT_FORESTNET_DIR
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(plan(args), indent=2, ensure_ascii=False), flush=True)
    if not args.execute:
        print("Plan only: add --execute to extract/convert features.", flush=True)
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    encoded_dir = materialize_upstream_encoded(args)
    output_dir, pca_dir = convert_encoded_dataset(
        encoded_dir, args.output_root, pca48=args.pca48
    )
    print(f"Saved raw features: {output_dir}", flush=True)
    if pca_dir is not None:
        print(f"Saved PCA48 features: {pca_dir}", flush=True)


if __name__ == "__main__":
    main()

