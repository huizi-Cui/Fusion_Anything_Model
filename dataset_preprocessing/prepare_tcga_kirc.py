"""TCGA-KIRC prepared omics tables and survival labels, without a learned encoder."""

from __future__ import annotations

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


DATA_ROOT = dataset_path('datasets/02_TCGA-KIRC/data/TCGA')
OUTPUT_DIR = ROOT / "data/tcga_kirc"
CANCER_TYPE = "KIRC"
SPLIT_SEED = 10
MODALITY_SPECS = (
    ("o1", "cnv.csv", "CNV"),
    ("o2", "meth.csv", "DNA methylation"),
    ("o3", "mirna.csv", "miRNA"),
    ("o4", "rnaseq.csv", "RNA-seq"),
    ("o5", "rppa.csv", "RPPA"),
    ("o6", "KIRC_cell_composition.csv", "cell composition"),
)


def _read_indexed(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    if not frame.index.is_unique:
        raise ValueError(f"Patient IDs are not unique in {path}")
    return frame


def main() -> None:
    data_dir = DATA_ROOT / CANCER_TYPE
    frames: dict[str, pd.DataFrame] = {}
    common_ids: pd.Index | None = None
    for modality, filename, _ in MODALITY_SPECS:
        frame = _read_indexed(data_dir / filename)
        frames[modality] = frame
        common_ids = frame.index if common_ids is None else common_ids.intersection(frame.index)

    survival_path = data_dir / "survival_clinical.csv"
    survival = pd.read_csv(survival_path).set_index("Patient Identifier")
    if not survival.index.is_unique or "Death" not in survival:
        raise ValueError(f"Invalid survival table: {survival_path}")
    assert common_ids is not None
    common_ids = common_ids.intersection(survival.index)

    arrays = {
        modality: frame.loc[common_ids].to_numpy(dtype=np.float32)
        for modality, frame in frames.items()
    }
    for modality, values in arrays.items():
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError(f"Invalid values in {CANCER_TYPE}/{modality}")
    death = pd.to_numeric(survival.loc[common_ids, "Death"], errors="coerce")
    if death.isna().any():
        raise ValueError(f"Invalid Death labels in {CANCER_TYPE}")
    labels = (death.to_numpy() > 0).astype(np.int64)

    indices = list(range(len(labels)))
    random.Random(SPLIT_SEED).shuffle(indices)
    test_size = len(indices) // 5
    validation_size = len(indices) // 4
    training_size = len(indices) - test_size - validation_size
    shuffled = np.asarray(indices, dtype=np.int64)
    train = shuffled[:training_size]
    validation = shuffled[training_size:training_size + validation_size]
    test = shuffled[training_size + validation_size:]
    context = np.concatenate([train, validation])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for modality, _, _ in MODALITY_SPECS:
        context_values, test_values = split_without_normalization(arrays[modality], context, test)
        np.save(OUTPUT_DIR / f"context_{modality}.npy", context_values)
        np.save(OUTPUT_DIR / f"test_{modality}.npy", test_values)
    patient_ids = common_ids.to_numpy(dtype=str)
    np.save(OUTPUT_DIR / "context_labels.npy", labels[context])
    np.save(OUTPUT_DIR / "test_labels.npy", labels[test])
    np.save(OUTPUT_DIR / "context_patient_ids.npy", patient_ids[context])
    np.save(OUTPUT_DIR / "test_patient_ids.npy", patient_ids[test])
    np.savez(OUTPUT_DIR / "split_indices.npz", train=train, validation=validation, context=context, test=test)

    modality_names = {modality: name for modality, _, name in MODALITY_SPECS}
    modality_dims = {modality: int(arrays[modality].shape[1]) for modality, _, _ in MODALITY_SPECS}
    info = {
        "dataset": "tcga_kirc",
        "display_name": "TCGA-KIRC",
        "cancer_type": CANCER_TYPE,
        "task": "binary survival-status classification",
        "n_classes": 2,
        "class_names": ["survived", "death"],
        "n_complete_case_patients": len(labels),
        "n_train": len(train),
        "n_validation": len(validation),
        "n_context": len(context),
        "n_test": len(test),
        "modalities": [modality for modality, _, _ in MODALITY_SPECS],
        "modality_names": modality_names,
        "modality_dims": modality_dims,
        "encoder": "identity mapping for six omics views",
        "label": "Death > 0 mapped to 1; Death == 0 mapped to 0",
        "patient_alignment": "intersection of all six modalities and survival labels",
        "split": "MultiBench datasets/TCGA seed-10 shuffled integer split; train+validation -> Context",
        "split_seed": SPLIT_SEED,
        "normalization": "none; raw encoder outputs retained without dataset-level normalization",
        "normalization_fit_scope": "none",
        "source_dir": str(data_dir.resolve()),
        "intermediate_projection": None,
    }
    with (OUTPUT_DIR / "info.pkl").open("wb") as file:
        pickle.dump(info, file)

    print(
        f"Saved TCGA-KIRC to {OUTPUT_DIR}: train={len(train)}, "
        f"validation={len(validation)}, context={len(context)}, "
        f"test={len(test)}, dims={modality_dims}"
    )


if __name__ == "__main__":
    main()

