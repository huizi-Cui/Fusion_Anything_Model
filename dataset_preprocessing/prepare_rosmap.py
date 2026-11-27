"""ROSMAP prepared omics CSVs with their original train and test splits."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
from .dataset_paths import dataset_path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from .encoders_zero import split_without_normalization

DATA_DIR = dataset_path('datasets/06_ROSMAP/data/ROSMAP')
OUTPUT_DIR = ROOT / "data/rosmap_zero"
MODALITIES = ["o1", "o2", "o3"]
CLASS_NAMES = ["class_0", "class_1"]


def main():
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    train_parts, test_parts = [], []
    for mod in MODALITIES:
        tr = pd.read_csv(DATA_DIR / f"{mod[-1]}_tr.csv", header=None).values.astype(np.float32)
        te = pd.read_csv(DATA_DIR / f"{mod[-1]}_te.csv", header=None).values.astype(np.float32)
        train_parts.append(tr)
        test_parts.append(te)
        print(f"  {mod}: train={tr.shape}, test={te.shape}")

    y_train = pd.read_csv(DATA_DIR / "labels_tr.csv", header=None).values.flatten().astype(np.int64)
    y_test = pd.read_csv(DATA_DIR / "labels_te.csv", header=None).values.flatten().astype(np.int64)
    n_cls = len(np.unique(np.concatenate([y_train, y_test])))
    print(f"ROSMAP zero: {n_cls} classes, "
          f"train={len(y_train)}, test={len(y_test)}, "
          f"dist={np.bincount(y_train)}")

    for i, mod in enumerate(MODALITIES):
        all_feat = np.concatenate([train_parts[i], test_parts[i]], axis=0)
        ctx_idx = np.arange(len(y_train))
        test_idx = np.arange(len(y_train), len(y_train) + len(y_test))
        context_values, test_values = split_without_normalization(all_feat, ctx_idx, test_idx)
        np.save(output_dir / f"context_{mod}.npy", context_values)
        np.save(output_dir / f"test_{mod}.npy", test_values)
        print(f"  {mod} raw split: context={context_values.shape}, test={test_values.shape}")

    np.save(output_dir / "context_labels.npy", y_train)
    np.save(output_dir / "test_labels.npy", y_test)
    np.save(output_dir / "context_indices.npy", np.arange(len(y_train), dtype=np.int64))
    np.savez(
        output_dir / "split_indices.npz",
        context=np.arange(len(y_train), dtype=np.int64),
        test=np.arange(len(y_train), len(y_train) + len(y_test), dtype=np.int64),
    )

    with open(output_dir / "info.pkl", "wb") as f:
        pickle.dump({
            "dataset": "rosmap_zero", "n_classes": n_cls,
            "class_names": CLASS_NAMES,
            "n_context": len(y_train), "n_test": len(y_test),
            "modalities": MODALITIES,
            "encoder": "identity (omics, zero-shot)",
            "normalization": "none; source feature values retained without dataset-level normalization",
            "normalization_fit_scope": "none",
        }, f)
    print(f"Done: {output_dir}")


if __name__ == "__main__":
    main()

