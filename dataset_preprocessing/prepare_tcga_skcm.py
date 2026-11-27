"""TCGA-SKCM prepared omics tables and survival labels, without a learned encoder."""

from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path
from .dataset_paths import dataset_path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .encoders_zero import split_without_normalization


DEFAULT_DATA_DIR = dataset_path('datasets/08_TCGA-SKCM/data/TCGA/SKCM')
DEFAULT_OUTPUT_DIR = ROOT / "data/tcga_skcm"
SPLIT_SEED = 10
MODALITY_FILES = (
    ("o1", "cnv.csv", "CNV"),
    ("o2", "meth.csv", "DNA methylation"),
    ("o3", "mirna.csv", "miRNA"),
    ("o4", "rnaseq.csv", "RNA-seq"),
    ("o5", "rppa.csv", "RPPA"),
    ("o6", "SKCM_cell_composition.csv", "cell composition"),
)


def load_aligned_data(
    data_dir: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, int]]:
    frames: dict[str, pd.DataFrame] = {}
    modality_dims: dict[str, int] = {}
    common_ids: pd.Index | None = None
    for modality, filename, _ in MODALITY_FILES:
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing TCGA-SKCM modality: {path}")
        frame = pd.read_csv(path, index_col=0)
        if not frame.index.is_unique:
            raise ValueError(f"Patient IDs are not unique in {path}")
        frames[modality] = frame
        modality_dims[modality] = int(frame.shape[1])
        common_ids = frame.index if common_ids is None else common_ids.intersection(frame.index)

    survival_path = data_dir / "survival_clinical.csv"
    survival = pd.read_csv(survival_path)
    required = {"Patient Identifier", "Death"}
    if not required.issubset(survival.columns):
        raise ValueError(f"Missing columns in {survival_path}: {sorted(required-set(survival.columns))}")
    survival = survival.set_index("Patient Identifier")
    if not survival.index.is_unique:
        raise ValueError("Patient IDs are not unique in survival_clinical.csv")
    assert common_ids is not None
    common_ids = common_ids.intersection(survival.index)
    if len(common_ids) < 2:
        raise ValueError("TCGA-SKCM has fewer than two complete-case patients")

    arrays: dict[str, np.ndarray] = {}
    for modality, frame in frames.items():
        values = frame.loc[common_ids].to_numpy(dtype=np.float32)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError(f"Non-finite or invalid values in {modality}")
        arrays[modality] = values
    death = pd.to_numeric(survival.loc[common_ids, "Death"], errors="coerce")
    if death.isna().any():
        raise ValueError("Invalid Death labels in complete-case patients")
    labels = (death.to_numpy() > 0).astype(np.int64)
    if len(np.unique(labels)) != 2:
        raise ValueError("TCGA-SKCM does not contain both survival-status classes")
    return arrays, labels, common_ids.to_numpy(dtype=str), modality_dims


def multibench_split(n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce datasets/TCGA/get_data.py's seed-10 shuffled split."""

    indices = list(range(n_samples))
    random.Random(SPLIT_SEED).shuffle(indices)
    test_size = n_samples // 5
    validation_size = n_samples // 4
    training_size = n_samples - test_size - validation_size
    values = np.asarray(indices, dtype=np.int64)
    training = values[:training_size]
    validation = values[training_size:training_size + validation_size]
    test = values[training_size + validation_size:]
    return training, validation, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    arrays, labels, patient_ids, modality_dims = load_aligned_data(data_dir)
    training, validation, test = multibench_split(len(labels))
    context = np.concatenate([training, validation])

    output_dir.mkdir(parents=True, exist_ok=True)
    context_relative = np.arange(len(context), dtype=np.int64)
    test_relative = np.arange(len(context), len(context) + len(test), dtype=np.int64)
    for modality, _, _ in MODALITY_FILES:
        ordered = np.concatenate([arrays[modality][context], arrays[modality][test]], axis=0)
        context_values, test_values = split_without_normalization(
            ordered, context_relative, test_relative
        )
        np.save(output_dir / f"context_{modality}.npy", context_values)
        np.save(output_dir / f"test_{modality}.npy", test_values)

    np.save(output_dir / "context_labels.npy", labels[context])
    np.save(output_dir / "test_labels.npy", labels[test])
    np.save(output_dir / "context_patient_ids.npy", patient_ids[context])
    np.save(output_dir / "test_patient_ids.npy", patient_ids[test])
    np.savez(
        output_dir / "split_indices.npz",
        train=training,
        validation=validation,
        context=context,
        test=test,
    )

    modality_names = {
        modality: public_name for modality, _, public_name in MODALITY_FILES
    }
    info = {
        "dataset": "tcga_skcm",
        "display_name": "TCGA-SKCM",
        "cancer_type": "SKCM",
        "task": "binary survival-status classification",
        "n_classes": 2,
        "class_names": ["survived", "death"],
        "n_complete_case_patients": len(labels),
        "n_train": len(training),
        "n_validation": len(validation),
        "n_context": len(context),
        "n_test": len(test),
        "modalities": [modality for modality, _, _ in MODALITY_FILES],
        "modality_names": modality_names,
        "modality_dims": modality_dims,
        "encoder": "identity mapping for six omics views",
        "label": "Death > 0 mapped to 1; Death == 0 mapped to 0",
        "patient_alignment": "intersection of all six modalities and survival labels",
        "split": "MultiBench datasets/TCGA seed-10 shuffled 55/25/20 integer split; train+validation -> Context",
        "split_seed": SPLIT_SEED,
        "normalization": "none; raw encoder outputs retained without dataset-level normalization",
        "normalization_fit_scope": "none",
        "source_dir": str(data_dir),
        "intermediate_projection": None,
    }
    with (output_dir / "info.pkl").open("wb") as file:
        pickle.dump(info, file)

    print(f"Saved TCGA-SKCM features to {output_dir}")
    print(
        f"train={len(training)}, validation={len(validation)}, "
        f"context={len(context)}, test={len(test)}"
    )
    print(
        f"Context labels={np.bincount(labels[context]).tolist()}, "
        f"Test labels={np.bincount(labels[test]).tolist()}"
    )
    print(f"Modality dimensions={modality_dims}")


if __name__ == "__main__":
    main()

