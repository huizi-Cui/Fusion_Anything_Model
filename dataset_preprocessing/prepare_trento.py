"""Trento HSI/DSM features use HyperFree-b and DOFA. Supply the encoded cache or the external extractor."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from .dataset_paths import dataset_path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


EXTRACT_ROOT = Path(__file__).resolve().parents[1]
if str(EXTRACT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTRACT_ROOT))

from .migration_common import save_dataset


EXTRACTOR_ROOT = Path("/path/to/feature_extractor")
EXTERNAL_DEPENDENCIES_ROOT = Path("/path/to/external_dependencies")
DEFAULT_RAW_ROOT = dataset_path('datasets/14_Trento/data')
DEFAULT_OUTPUT_ROOT = EXTRACT_ROOT / "data"
DEFAULT_ENCODED_ROOT = DEFAULT_OUTPUT_ROOT / "_encoded_cache"

SOURCE_DATASET = "trento"
SOURCE_OUTPUT_NAME = "trento_remote_hyperfree_b_auxdofa_full_712_seed42_encoded"
OUTPUT_NAME = "trento_remote_hyperfree_b_auxdofa_full_712_seed42"
EXPECTED_MODALITIES = ("hsi", "dsm")
EXPECTED_KEPT_LABELS = (1, 2, 3, 4, 5, 6)
EXPECTED_DROPPED_LABELS: tuple[int, ...] = ()
EXPECTED_LABEL_MAP = {
    str(original): mapped for mapped, original in enumerate(EXPECTED_KEPT_LABELS)
}
EXPECTED_SOURCE_POOL = (
    "complete labelled map; ignores loader-generated 100-per-class split"
)
SEED = 42


def load_info(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected dict metadata in {path}, got {type(value)!r}")
    return value


def validate_encoder_contract(info: dict[str, Any]) -> None:
    detail = info.get("encoder_detail", {})
    hsi = detail.get("hsi", {})
    dsm = detail.get("dsm", {})
    if hsi.get("name") != "hyperfree_b" or hsi.get("frozen") is not True:
        raise ValueError(f"Trento HSI encoder drift: {hsi!r}")
    if int(hsi.get("source_patch_size", -1)) != 5:
        raise ValueError(f"Trento HSI patch-size drift: {hsi!r}")
    if dsm.get("name") != "DOFA_ViT_base_e100" or dsm.get("frozen") is not True:
        raise ValueError(f"Trento DSM encoder drift: {dsm!r}")
    wavelengths = [float(value) for value in dsm.get("wavelengths_um", [])]
    if wavelengths != [0.55, 0.55, 0.55]:
        raise ValueError(f"Trento DSM pseudo-wavelength drift: {wavelengths!r}")


def validate_encoded_contract(encoded_dir: Path, info: dict[str, Any]) -> None:
    """Validate every frozen upstream decision before downstream processing."""

    if info.get("source_dataset") != SOURCE_DATASET:
        raise ValueError(
            f"Wrong source dataset in {encoded_dir}: {info.get('source_dataset')!r}"
        )
    if info.get("source_pool") != EXPECTED_SOURCE_POOL:
        raise ValueError(f"Trento source-pool drift: {info.get('source_pool')!r}")
    if tuple(info.get("modalities", ())) != EXPECTED_MODALITIES:
        raise ValueError(
            f"Wrong modality order in {encoded_dir}: {info.get('modalities')!r}"
        )
    if info.get("label_map") != EXPECTED_LABEL_MAP:
        raise ValueError(f"Trento class mapping drift: {info.get('label_map')!r}")
    dropped = tuple(sorted(int(value) for value in info.get("dropped_classes", {})))
    if dropped != EXPECTED_DROPPED_LABELS:
        raise ValueError(
            "Trento must retain all six classes; unexpected dropped classes: "
            f"{dropped}"
        )
    if int(info.get("n_classes", -1)) != 6:
        raise ValueError(f"Expected six retained classes in {encoded_dir}")
    validate_encoder_contract(info)

    split = info.get("split", {})
    policy = str(split.get("policy", ""))
    if split.get("seed") != SEED or "7:1:2" not in policy:
        raise ValueError(f"Unexpected split contract in {encoded_dir}: {split!r}")
    counts = split.get("counts", {})
    if set(counts) != {"train", "validation", "test", "pool"}:
        raise ValueError(f"Incomplete split counts in {encoded_dir}: {counts!r}")
    if sum(int(counts[name]) for name in ("train", "validation", "test")) != int(
        counts["pool"]
    ):
        raise ValueError(f"Split is not exhaustive in {encoded_dir}: {counts!r}")

    split_pool_indices: list[np.ndarray] = []
    dimensions: dict[str, int] = {}
    expected_classes = np.arange(6, dtype=np.int64)
    for split_name in ("train", "validation", "test"):
        label_path = encoded_dir / f"{split_name}_labels.npy"
        index_path = encoded_dir / f"{split_name}_pool_indices.npy"
        if not label_path.is_file() or not index_path.is_file():
            raise FileNotFoundError(
                f"Missing frozen {split_name} membership in {encoded_dir}"
            )
        labels = np.load(label_path, mmap_mode="r")
        indices = np.load(index_path, mmap_mode="r")
        if labels.shape != indices.shape or len(labels) != int(counts[split_name]):
            raise ValueError(f"{split_name} membership/label mismatch in {encoded_dir}")
        if not np.array_equal(np.unique(labels), expected_classes):
            raise ValueError(f"{split_name} does not contain all six classes")
        split_pool_indices.append(np.asarray(indices, dtype=np.int64))
        for modality in EXPECTED_MODALITIES:
            path = encoded_dir / f"{split_name}_{modality}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"Missing encoded feature matrix: {path}")
            values = np.load(path, mmap_mode="r")
            if values.ndim != 2 or len(values) != len(labels):
                raise ValueError(f"Invalid encoded shape for {path}: {values.shape}")
            if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
                raise ValueError(f"Invalid encoded values in {path}")
            previous = dimensions.setdefault(modality, int(values.shape[1]))
            if int(values.shape[1]) != previous:
                raise ValueError(f"{modality} dimension differs across splits")
    metadata_dimensions = {
        str(key): int(value) for key, value in info.get("dimensions", {}).items()
    }
    if dimensions != metadata_dimensions:
        raise ValueError(
            f"Encoded dimension metadata drift: arrays={dimensions}, "
            f"info={metadata_dimensions}"
        )

    joined = np.concatenate(split_pool_indices)
    if not np.array_equal(
        np.sort(joined), np.arange(int(counts["pool"]), dtype=np.int64)
    ):
        raise ValueError(f"Frozen split membership is not disjoint/exhaustive: {encoded_dir}")


def materialize_upstream_encoded(args: argparse.Namespace) -> Path:
    """Run exact encoder stage once, or reuse its validated cache."""

    encoded_dir = args.encoded_root / SOURCE_OUTPUT_NAME
    incomplete = encoded_dir.with_name(encoded_dir.name + ".incomplete")
    if incomplete.exists():
        raise RuntimeError(
            f"Incomplete upstream extraction exists: {incomplete}. Inspect it explicitly."
        )
    if not encoded_dir.is_dir():
        if not args.extractor_root.is_dir():
            raise FileNotFoundError(f"External extractor repository is missing: {args.extractor_root}")
        sys.path.insert(0, str(args.extractor_root))
        sys.path.insert(0, str(EXTERNAL_DEPENDENCIES_ROOT))
        from inference_multimodal_1024d import prepare_remote_full_712 as upstream

        args.encoded_root.mkdir(parents=True, exist_ok=True)
        upstream_args = SimpleNamespace(
            output_base=args.encoded_root,
            raw_root=args.raw_root,
            forestnet_data_dir=dataset_path("datasets/15_ForestNet/data/ForestNetDataset"),
            batch_size=args.batch_size,
            dofa_batch_size=args.dofa_batch_size,
            hsi_gsd_meters=args.hsi_gsd_meters,
        )
        produced = upstream.prepare_one(SOURCE_DATASET, upstream_args)
        if produced.resolve() != encoded_dir.resolve():
            raise RuntimeError(f"Unexpected upstream output: {produced} vs {encoded_dir}")
    info = load_info(encoded_dir / "info.pkl")
    validate_encoded_contract(encoded_dir, info)
    return encoded_dir


def convert_encoded_dataset(
    encoded_dir: Path,
    output_root: Path,
    *,
    pca48: bool,
) -> tuple[Path, Path | None]:
    """Construct Context/Test and save raw features and optionally create PCA48."""

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
        display_name="Trento",
        n_train=n_train,
        n_validation=n_validation,
        n_context=n_context,
        n_test=n_test,
        context_construction="ordered upstream train followed by validation",
        context_source_splits=["train", "validation"],
        test_source_split="test",
        structured_modalities=[],
        upstream_encoded_dir=str(encoded_dir.resolve()),
        upstream_responsibilities=info.get("responsibilities", []),
        upstream_excluded_operations=info.get("excluded_operations", []),
        downstream_operations=[
            "train+validation Context construction",
            "no dataset-level normalization",
            *(["Context-only PCA48"] if pca48 else []),
        ],
        split_note=(
            "The complete labelled map is repartitioned with seed 42; the "
            "loader-generated 100-per-class training split is not used."
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
        "dataset": "Trento",
        "source_pool": EXPECTED_SOURCE_POOL,
        "class_policy": {
            "kept_original_labels": list(EXPECTED_KEPT_LABELS),
            "dropped_original_labels": [],
            "mapped_labels": "0..5",
            "note": "keep all six classes; no semantic merge",
        },
        "upstream_split": "stratified train/validation/test=7:1:2, seed=42",
        "context": "ordered train+validation",
        "test": "upstream test",
        "encoders": {"hsi": "frozen HyperFree-b", "dsm": "frozen DOFA"},
        "normalization": "none; raw encoder outputs retained without dataset-level normalization",
        "pca48": bool(args.pca48),
        "pca_fit_scope": "Context",
        "encoded_cache": str((args.encoded_root / SOURCE_OUTPUT_NAME).resolve()),
        "output": str((args.output_root / OUTPUT_NAME).resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--pca48", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--encoded-root", type=Path, default=DEFAULT_ENCODED_ROOT)
    parser.add_argument("--extractor-root", type=Path, default=EXTRACTOR_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dofa-batch-size", type=int, default=64)
    parser.add_argument("--hsi-gsd-meters", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(plan(args), indent=2, ensure_ascii=False), flush=True)
    if not args.execute:
        print("Plan only: add --execute to extract/convert features.", flush=True)
        return
    if args.device != "cuda":
        raise ValueError("frozen HyperFree-b/DOFA contract is pinned to CUDA")
    cache = args.encoded_root / SOURCE_OUTPUT_NAME
    if not cache.is_dir() and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to materialize the upstream encoder cache")
    encoded_dir = materialize_upstream_encoded(args)
    output_dir, pca_dir = convert_encoded_dataset(
        encoded_dir, args.output_root, pca48=args.pca48
    )
    print(f"Saved raw features: {output_dir}", flush=True)
    if pca_dir is not None:
        print(f"Saved PCA48 features: {pca_dir}", flush=True)


if __name__ == "__main__":
    main()

