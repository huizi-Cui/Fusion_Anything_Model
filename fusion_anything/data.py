"""Prepared-feature loading and Context-only projection validation."""
from __future__ import annotations
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np


@dataclass(frozen=True)
class GroupData:
    """Fully materialized TabPFN input for one group."""

    group_code: str
    group_name: str
    data_dir: Path
    context_x: np.ndarray
    context_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    feature_blocks: tuple[dict[str, Any], ...]
    test_feature_blocks: tuple[dict[str, Any], ...]
    info: dict[str, Any]

    @property
    def n_context(self) -> int:
        """Number of labeled Context rows."""
        return int(self.context_x.shape[0])

    @property
    def n_test(self) -> int:
        """Number of Query rows."""
        return int(self.test_x.shape[0])

    @property
    def n_features(self) -> int:
        """Total width of the concatenated modality features."""
        return int(self.context_x.shape[1])

    @property
    def n_classes(self) -> int:
        """Number of distinct classes represented in Context."""
        return int(len(np.unique(self.context_y)))


def sha256_indices(indices: np.ndarray) -> str:
    """Hash a sequence after conversion to canonical int64 bytes."""
    values = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def label_distribution(labels: np.ndarray) -> dict[str, int]:
    """Return the count for each integer label as JSON-compatible values."""
    classes, counts = np.unique(np.asarray(labels), return_counts=True)
    return {str(int(cls)): int(count) for cls, count in zip(classes, counts)}


def load_registry(path: Path) -> tuple[Path, list[dict[str, Any]]]:
    """Read the prepared-feature registry and reject duplicate task names."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = Path(payload["data_root"]).resolve()
    entries = payload["datasets"]
    names = [str(entry["name"]) for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate dataset names in registry")
    return root, entries


def _load_labels(path: Path) -> np.ndarray:
    labels = np.asarray(np.load(path, mmap_mode="r"))
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Expected one-dimensional integer labels: {path}")
    return labels.astype(np.int64, copy=False)


def load_pca_dataset(data_root: Path, entry: dict[str, Any]) -> GroupData:
    """Load and validate one prepared Context/Query dataset.
    
    Args:
        data_root: Parent directory used when entry has no explicit data_dir.
        entry: Task name, ordered modalities, and structured modality declarations.
    
    Returns:
        Materialized float32 features, integer labels, and projection metadata.
    
    Raises:
        ValueError: Shapes, labels, finite values, or Context-only PCA violate
            the prepared-feature contract.
        OSError: A required input file cannot be read.
    
    Notes:
        Pickle metadata must come from a trusted prepared-feature pipeline.
        Query labels are loaded for evaluation, not passed to model prediction."""
    data_dir = Path(entry.get("data_dir", data_root / str(entry["name"]))).resolve()
    with (data_dir / "info.pkl").open("rb") as file:
        info = pickle.load(file)
    if info.get("projection") != "pca" or int(info.get("target_dim", -1)) != 48:
        raise ValueError(f"{data_dir}: expected projection=pca and target_dim=48")

    modalities = [str(value) for value in entry["modalities"]]
    if len(modalities) > 8:
        raise ValueError(f"{data_dir}: more than 8 modalities")
    # Projection metadata is authoritative for the automatic <48d structured
    # rule; registry declarations are retained for backward compatibility.
    structured = set(entry.get("structured_modalities") or []) | set(
        info.get("structured_modalities") or []
    )
    unknown_structured = structured.difference(modalities)
    if unknown_structured:
        raise ValueError(
            f"Unknown structured modalities in {data_dir}: {sorted(unknown_structured)}"
        )
    context_arrays: list[np.ndarray] = []
    test_arrays: list[np.ndarray] = []
    context_blocks: list[dict[str, Any]] = []
    test_blocks: list[dict[str, Any]] = []
    start = 0
    n_context = n_test = None
    for modality in modalities:
        context_path = data_dir / f"context_{modality}.npy"
        test_path = data_dir / f"test_{modality}.npy"
        context = np.asarray(np.load(context_path, mmap_mode="r"), dtype=np.float32)
        test = np.asarray(np.load(test_path, mmap_mode="r"), dtype=np.float32)
        if context.ndim != 2 or test.ndim != 2 or context.shape[1] != test.shape[1]:
            raise ValueError(f"Invalid feature shape for {data_dir}/{modality}")
        if not np.isfinite(context).all() or not np.isfinite(test).all():
            raise ValueError(f"Non-finite values in {data_dir}/{modality}")
        if modality not in structured and context.shape[1] > 48:
            raise ValueError(f"Unstructured modality is not PCA48: {data_dir}/{modality}")
        n_context = len(context) if n_context is None else n_context
        n_test = len(test) if n_test is None else n_test
        if len(context) != n_context or len(test) != n_test:
            raise ValueError(f"Row mismatch in {data_dir}")
        end = start + int(context.shape[1])
        common = {
            "name": modality,
            "shape": [len(context), int(context.shape[1])],
            "source_dtype": str(context.dtype),
            "column_start": start,
            "column_end_exclusive": end,
            "nan_count": 0,
        }
        context_blocks.append({**common, "path": str(context_path.resolve())})
        test_blocks.append({
            **common,
            "shape": [len(test), int(test.shape[1])],
            "path": str(test_path.resolve()),
        })
        context_arrays.append(context)
        test_arrays.append(test)
        start = end

    context_y = _load_labels(data_dir / "context_labels.npy")
    test_y = _load_labels(data_dir / "test_labels.npy")
    if len(context_y) != n_context or len(test_y) != n_test:
        raise ValueError(f"Feature/label row mismatch in {data_dir}")
    classes = np.unique(np.concatenate([context_y, test_y]))
    if not np.array_equal(classes, np.arange(len(classes), dtype=np.int64)):
        raise ValueError(f"Non-contiguous labels in {data_dir}: {classes.tolist()}")
    if len(classes) > 10 or not set(classes).issubset(set(np.unique(context_y))):
        raise ValueError(f"Unsupported class space in {data_dir}")

    unstructured = [modality for modality in modalities if modality not in structured]
    pca_models = {}
    if unstructured:
        model_path = data_dir / "projection_models.pkl"
        with model_path.open("rb") as file:
            pca_models = pickle.load(file)
    for modality in modalities:
        if modality in structured:
            continue
        model = pca_models.get(modality)
        if model is None:
            raise ValueError(f"Missing PCA model for {data_dir}/{modality}")
        if int(getattr(model, "n_samples_", -1)) != n_context:
            raise ValueError(f"PCA for {data_dir}/{modality} was not fitted on Context")
        expected = int(context_arrays[modalities.index(modality)].shape[1])
        fitted_components = int(getattr(model, "n_components_", -1))
        if fitted_components <= 0 or fitted_components > expected:
            raise ValueError(f"PCA component mismatch for {data_dir}/{modality}")
        if fitted_components < expected:
            index = modalities.index(modality)
            if (
                np.any(context_arrays[index][:, fitted_components:] != 0)
                or np.any(test_arrays[index][:, fitted_components:] != 0)
            ):
                raise ValueError(f"Nonzero PCA padding for {data_dir}/{modality}")

    info = dict(info)
    info["structured_modalities"] = sorted(structured)
    info["pca_fit_scope"] = "context"
    return GroupData(
        group_code="TC",
        group_name="tabpfn_v2_pca48",
        data_dir=data_dir,
        context_x=np.concatenate(context_arrays, axis=1),
        context_y=context_y,
        test_x=np.concatenate(test_arrays, axis=1),
        test_y=test_y,
        feature_blocks=tuple(context_blocks),
        test_feature_blocks=tuple(test_blocks),
        info=info,
    )
