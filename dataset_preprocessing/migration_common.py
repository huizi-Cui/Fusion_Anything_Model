"""Save Context and Test features and optionally run PCA."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .encoders_zero import split_without_normalization
from .prepare_projection import project_dataset


def save_dataset(output_dir: Path, labels: np.ndarray, context: np.ndarray,
                 test: np.ndarray, features: dict[str, np.ndarray], info: dict,
                 split_indices: dict[str, np.ndarray], pca48: bool) -> None:
    """Save raw feature splits without dataset-level normalization and optionally create PCA48."""
    output_dir.mkdir(parents=True, exist_ok=True)
    context = np.asarray(context, dtype=np.int64)
    test = np.asarray(test, dtype=np.int64)
    np.save(output_dir / "context_labels.npy", labels[context].astype(np.int64))
    np.save(output_dir / "test_labels.npy", labels[test].astype(np.int64))
    np.save(output_dir / "context_indices.npy", context)
    np.savez(output_dir / "split_indices.npz", **split_indices)
    dims = {}
    for name, values in features.items():
        values = np.asarray(values, dtype=np.float32)
        if values.shape[0] != len(labels):
            raise ValueError(f"{name}: {values.shape[0]} rows for {len(labels)} labels")
        context_values, test_values = split_without_normalization(values, context, test)
        np.save(output_dir / f"context_{name}.npy", context_values)
        np.save(output_dir / f"test_{name}.npy", test_values)
        dims[name] = int(values.shape[1])
    metadata = dict(info)
    metadata.update(
        dataset=output_dir.name,
        n_context=len(context), n_test=len(test), modality_dims=dims,
        normalization="none; raw encoder outputs retained without dataset-level normalization",
        normalization_fit_scope="none",
    )
    with (output_dir / "info.pkl").open("wb") as handle:
        pickle.dump(metadata, handle)
    if pca48:
        print(project_dataset(output_dir.name, source_root=output_dir.parent,
                              output_root=output_dir.parent), flush=True)

