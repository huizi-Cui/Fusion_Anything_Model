"""Prepare Context-only PCA features and persist their projection metadata."""
from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
TABULAR_MODALITIES = {"tabular", "static", "timeseries"}


def pad_right(values: np.ndarray, target_dim: int) -> np.ndarray:
    """Convert features to float32 and pad the right side with zero columns."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] > target_dim:
        raise ValueError(f"Cannot pad shape {values.shape} to {target_dim} columns")
    if values.shape[1] == target_dim:
        return values
    return np.pad(values, ((0, 0), (0, target_dim - values.shape[1]))).astype(
        np.float32, copy=False
    )


def load_context_indices(source: Path, n_context: int) -> np.ndarray:
    """Read unique original Context row indices from split metadata."""
    direct = source / "context_indices.npy"
    split = source / "split_indices.npz"
    if direct.is_file():
        indices = np.load(direct)
    elif split.is_file():
        with np.load(split) as payload:
            if "context" not in payload:
                raise ValueError(f"Missing context indices in {split}")
            indices = payload["context"]
    else:
        raise FileNotFoundError(f"Missing original Context indices in {source}")
    indices = np.asarray(indices, dtype=np.int64)
    if indices.shape != (n_context,) or len(np.unique(indices)) != n_context:
        raise ValueError(f"Invalid Context indices in {source}: {indices.shape}")
    return indices


def modality_names(source: Path, info: dict) -> list[str]:
    """Read modality names from metadata or infer them from feature filenames."""
    names = list(info.get("modality_dims", {}))
    if names:
        return names
    metadata_stems = {
        "context_labels",
        "context_indices",
        "context_patient_ids",
        "context_sample_ids",
        "context_utterance_ids",
    }
    return [
        path.stem.removeprefix("context_")
        for path in sorted(source.glob("context_*.npy"))
        if path.stem not in metadata_stems
    ]


def project_dataset(
    dataset: str,
    pca_dim: int = 48,
    *,
    source_root: Path | None = None,
    output_root: Path | None = None,
) -> Path:
    """Prepare disk-backed features with Context-only PCA and saved metadata.
    
    Args:
        dataset: Source task directory name.
        pca_dim: Target dimension for unstructured modality features.
        source_root: Parent of the source task directory.
        output_root: Parent in which to write the projected task directory.
    
    Returns:
        Directory containing the projected features and fitted PCA objects.
    
    Notes:
        Declared structured, named tabular, and narrower-than-target modalities
        retain their original columns. PCA is fitted in ascending original
        Context index order; Query features only use the fitted transform."""
    source = (source_root or DATA_ROOT) / dataset
    output = (output_root or DATA_ROOT) / f"{dataset}_pca"
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {source}")

    with (source / "info.pkl").open("rb") as file:
        info = pickle.load(file)
    context_labels = np.load(source / "context_labels.npy")
    test_labels = np.load(source / "test_labels.npy")
    context_indices = load_context_indices(source, len(context_labels))
    fit_positions = np.argsort(context_indices, kind="stable")
    output.mkdir(parents=True, exist_ok=True)

    models = {}
    copied_modalities = []
    declared_structured = set(info.get("structured_modalities", ()))
    for modality in modality_names(source, info):
        context_path = source / f"context_{modality}.npy"
        test_path = source / f"test_{modality}.npy"
        context = np.load(context_path)
        test = np.load(test_path)
        if context.ndim != 2 or test.ndim != 2 or context.shape[1] != test.shape[1]:
            raise ValueError(f"Invalid feature shape for {source}/{modality}")

        # Modalities narrower than PCA48 are treated as structured/tabular:
        # retain their original columns and route them through TabPFN's native
        # structured encoder rather than MGM+CAP.  Explicitly declared
        # structured modalities remain structured regardless of width.
        is_structured = (
            modality in TABULAR_MODALITIES
            or modality in declared_structured
            or context.shape[1] < pca_dim
        )
        if is_structured:
            shutil.copy2(context_path, output / context_path.name)
            shutil.copy2(test_path, output / test_path.name)
            copied_modalities.append(modality)
            continue

        components = min(pca_dim, context.shape[0], context.shape[1])
        model = PCA(n_components=components, svd_solver="auto", random_state=42)
        canonical_context = np.ascontiguousarray(context[fit_positions])
        canonical_projected = pad_right(
            model.fit_transform(canonical_context).astype(np.float32), pca_dim
        )
        test_projected = pad_right(
            model.transform(test).astype(np.float32), pca_dim
        )
        context_projected = np.empty_like(canonical_projected)
        context_projected[fit_positions] = canonical_projected
        np.save(output / context_path.name, context_projected)
        np.save(output / test_path.name, test_projected)
        models[modality] = model

    np.save(output / "context_labels.npy", context_labels)
    np.save(output / "test_labels.npy", test_labels)
    np.save(output / "context_indices.npy", context_indices)
    if models:
        with (output / "projection_models.pkl").open("wb") as file:
            pickle.dump(models, file)

    info.update(
        dataset=f"{dataset}_pca",
        projection="pca",
        target_dim=pca_dim,
        tabular_kept=copied_modalities,
        structured_modalities=copied_modalities,
        structured_dimension_rule=f"pre-PCA modality dimension < {pca_dim}",
        pca_fit_scope="context",
        pca_fit_order="ascending_original_context_index",
        pca_output_padding="right_zero_to_target_dim",
    )
    with (output / "info.pkl").open("wb") as file:
        pickle.dump(info, file)
    return output


def main() -> None:
    """Project one prepared dataset using the command-line configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--projection", choices=["pca"], default="pca")
    parser.add_argument("--pca_dim", type=int, default=48)
    args = parser.parse_args()
    print(project_dataset(args.dataset, args.pca_dim))


if __name__ == "__main__":
    main()
