"""Extract Derm7pt features and fit PCA48 using Context samples only."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from fusion_anything.preprocessing import projection
from fusion_anything.preprocessing.image_encoder import (
    build_resnet18_encoder,
    extract_resnet18_features,
    split_without_normalization,
)

DATASET_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = DATASET_DIR / "raw"
DEFAULT_OUTPUT_ROOT = DATASET_DIR / "processed"
STRUCTURED_COLS = [
    "seven_point_score", "pigment_network", "streaks", "pigmentation",
    "regression_structures", "dots_and_globules", "blue_whitish_veil",
    "vascular_structures", "level_of_diagnostic_difficulty", "elevation",
    "location", "sex", "management",
]
# Fixed mapping from the 20 diagnosis strings to the seven evaluation classes.
LABEL_20_TO_7 = {
    "melanoma (in situ)": 0, "melanoma (less than 0.76 mm)": 0,
    "melanoma (0.76 to 1.5 mm)": 0, "melanoma (more than 1.5 mm)": 0,
    "melanoma metastasis": 0, "melanoma": 0,
    "basal cell carcinoma": 1,
    "clark nevus": 2,
    "reed or spitz nevus": 3,
    "dermal nevus": 4, "blue nevus": 4, "combined nevus": 4,
    "congenital nevus": 4, "recurrent nevus": 4,
    "seborrheic keratosis": 5,
    "vascular lesion": 6, "dermatofibroma": 6, "lentigo": 6,
    "melanosis": 6, "miscellaneous": 6,
}
CLASS_NAMES = [
    "melanoma",
    "basal_cell_carcinoma",
    "clark_nevus",
    "reed_or_spitz_nevus",
    "other_nevus",
    "seborrheic_keratosis",
    "other_lesion",
]


def encode_structured(
    df: pd.DataFrame, columns: list[str], context_idx: np.ndarray,
) -> np.ndarray:
    """Encode categorical columns using an encoder fitted only on Context."""
    df = df.copy()
    num_cols = df[columns].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in columns if c not in num_cols]
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(df.iloc[context_idx][cat_cols].astype(str))
        df[cat_cols] = enc.transform(df[cat_cols].astype(str))
    return df[columns].values.astype(np.float32)


def extract_raw_features(data_dir: Path, output_dir: Path) -> None:
    """Generate frozen image features and Context-fitted table encodings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(data_dir / "meta" / "meta.csv")
    img_dir = data_dir / "images"

    # Label mapping: 20 diagnosis strings → 7 classes
    mapped_labels = meta["diagnosis"].map(LABEL_20_TO_7)
    if mapped_labels.isna().any():
        unknown = sorted(meta.loc[mapped_labels.isna(), "diagnosis"].astype(str).unique())
        raise ValueError(f"Unsupported Derm7pt diagnosis labels: {unknown}")
    y_all = mapped_labels.to_numpy(dtype=np.int64)
    n_cls = len(set(y_all))
    print(f"Derm7pt zero: {n_cls} classes, {len(meta)} samples, dist={np.bincount(y_all)}")

    # Split FIRST: train+valid → context, test → test (避免数据泄露)
    meta_dir = data_dir / "meta"
    train_idx = pd.read_csv(meta_dir / "train_indexes.csv").values.flatten()
    valid_idx = pd.read_csv(meta_dir / "valid_indexes.csv").values.flatten()
    test_idx_csv = pd.read_csv(meta_dir / "test_indexes.csv").values.flatten()
    ctx_mask = np.zeros(len(meta), dtype=bool)
    ctx_mask[train_idx] = True
    ctx_mask[valid_idx] = True
    test_mask = np.zeros(len(meta), dtype=bool)
    test_mask[test_idx_csv] = True
    ctx_idx = np.where(ctx_mask)[0]
    test_idx = np.where(test_mask)[0]
    print(f"  Context: {len(ctx_idx)}, Test: {len(test_idx)}")

    # 保留各模态的原始特征，不执行数据集级归一化。
    image_paths = [str(img_dir / row["clinic"]) for _, row in meta.iterrows()]
    encoder = build_resnet18_encoder(device)
    img_feat = extract_resnet18_features(encoder, image_paths, device)
    img_ctx, img_test = split_without_normalization(img_feat, ctx_idx, test_idx)
    print(f"  Image (resnet18): context={img_ctx.shape}, test={img_test.shape}")

    # Fit categorical encodings on Context only; retain numeric values without normalization.
    tabular = encode_structured(meta, STRUCTURED_COLS, ctx_idx)
    tab_ctx, tab_test = split_without_normalization(tabular, ctx_idx, test_idx)
    print(f"  Tabular: context={tab_ctx.shape}, test={tab_test.shape}")

    np.save(output_dir / "context_image.npy", img_ctx)
    np.save(output_dir / "test_image.npy", img_test)
    np.save(output_dir / "context_tabular.npy", tab_ctx)
    np.save(output_dir / "test_tabular.npy", tab_test)
    np.save(output_dir / "context_labels.npy", y_all[ctx_idx])
    np.save(output_dir / "test_labels.npy", y_all[test_idx])
    np.savez(output_dir / "split_indices.npz", context=ctx_idx, test=test_idx)

    with (output_dir / "info.pkl").open("wb") as file:
        pickle.dump(
            {
                "dataset": "derm7pt_zero",
                "n_classes": n_cls,
                "class_names": CLASS_NAMES,
                "n_context": len(ctx_idx),
                "n_test": len(test_idx),
                "encoder": "resnet18 + encoder_x (zero-shot)",
                "normalization": (
                    "none; raw encoder outputs retained without dataset-level "
                    "normalization"
                ),
                "normalization_fit_scope": "none",
            },
            file,
        )
    print(f"Done: {output_dir}")


def main() -> int:
    """Validate raw inputs, extract features, and create the PCA48 task directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help="raw Derm7pt directory (default: datasets/Derm7pt/raw)",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help="feature output root (default: datasets/Derm7pt/processed)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    required = [data_dir / "images"] + [
        data_dir / "meta" / name for name in (
            "meta.csv", "train_indexes.csv", "valid_indexes.csv", "test_indexes.csv"
        )
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("Missing Derm7pt inputs:\n" + "\n".join(missing))

    extract_raw_features(data_dir, output_root / "derm7pt_zero")
    result = projection.project_dataset(
        "derm7pt_zero", pca_dim=48,
        source_root=output_root, output_root=output_root,
    )
    print(f"PCA48 ready: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
