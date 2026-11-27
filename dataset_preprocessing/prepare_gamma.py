"""GAMMA fundus and sampled OCT slices encoded with ImageNet ResNet-18."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
from .dataset_paths import dataset_path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from .encoders_zero import (
    build_resnet18_encoder, extract_resnet18_features,
    split_without_normalization,
)

DATA_DIR = dataset_path('datasets/01_GAMMA/data/GAMMA')
OUTPUT_DIR = ROOT / "data/gamma_zero"
CLASS_NAMES = ["non", "early", "mid_advanced"]
OCT_SLICE_COUNT = 16  # 每个患者采样16张OCT切片取均值


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载标签 (one-hot → single label)
    gt = pd.read_excel(DATA_DIR / "training" / "glaucoma_grading_training_GT.xlsx")
    class_cols = ["non", "early", "mid_advanced"]
    label_map = {c: i for i, c in enumerate(class_cols)}
    y_all = gt[class_cols].idxmax(axis=1).map(label_map).values.astype(np.int64)
    n_cls = len(class_cols)
    print(f"GAMMA zero: {n_cls} classes, {len(gt)} samples, "
          f"dist={np.bincount(y_all)}")

    # 划分 context/test (80/20, 分层)
    all_idx = np.arange(len(gt))
    ctx_idx, test_idx = train_test_split(
        all_idx, test_size=0.2, random_state=42, stratify=y_all)
    print(f"  Context: {len(ctx_idx)}, Test: {len(test_idx)}")
    print(f"  Context dist: {np.bincount(y_all[ctx_idx])}")
    print(f"  Test dist: {np.bincount(y_all[test_idx])}")

    img_base = DATA_DIR / "training" / "multi-modality_images"
    encoder = build_resnet18_encoder(device)
    import tempfile, shutil
    from PIL import Image as PILImage
    tmp_dir = Path(tempfile.mkdtemp())


    print("  Extracting fundus features...")
    fundus_paths = []
    for idx in range(len(gt)):
        pid = f"{idx+1:04d}"
        img_path = img_base / pid / f"{pid}.jpg"
        try:
            pil_img = PILImage.open(img_path).convert('RGB').resize((224, 224))
        except:
            pil_img = PILImage.new('RGB', (224, 224))
        tmp_path = tmp_dir / f"fundus_{pid}.png"
        pil_img.save(tmp_path)
        fundus_paths.append(str(tmp_path))
    fundus_feat = extract_resnet18_features(encoder, fundus_paths, device)
    fundus_ctx, fundus_test = split_without_normalization(
        fundus_feat, ctx_idx, test_idx)
    print(f"  Fundus: context={fundus_ctx.shape}, test={fundus_test.shape}")


    print(f"  Extracting OCT features (sampling {OCT_SLICE_COUNT} slices/patient)...")
    oct_feats = []
    for idx in range(len(gt)):
        pid = f"{idx+1:04d}"
        oct_dir = img_base / pid / pid
        # 采样OCT切片 (0, 1, ..., 255中的均匀采样)
        slice_indices = np.linspace(0, 255, OCT_SLICE_COUNT).astype(int)
        slice_paths = []
        for si in slice_indices:
            slice_file = oct_dir / f"{si}_image.jpg"
            try:
                pil_img = PILImage.open(slice_file).convert('RGB').resize((224, 224))
            except:
                pil_img = PILImage.new('RGB', (224, 224))
            tmp_path = tmp_dir / f"oct_{pid}_{si}.png"
            pil_img.save(tmp_path)
            slice_paths.append(str(tmp_path))
        # 提取所有切片的特征并取均值
        slice_feat = extract_resnet18_features(encoder, slice_paths, device)
        oct_feats.append(slice_feat.mean(axis=0))
    oct_feat = np.stack(oct_feats, axis=0)
    oct_ctx, oct_test = split_without_normalization(oct_feat, ctx_idx, test_idx)
    print(f"  OCT: context={oct_ctx.shape}, test={oct_test.shape}")

    shutil.rmtree(tmp_dir)

    np.save(output_dir / "context_fundus.npy", fundus_ctx)
    np.save(output_dir / "test_fundus.npy", fundus_test)
    np.save(output_dir / "context_oct.npy", oct_ctx)
    np.save(output_dir / "test_oct.npy", oct_test)
    np.save(output_dir / "context_labels.npy", y_all[ctx_idx])
    np.save(output_dir / "test_labels.npy", y_all[test_idx])
    np.savez(output_dir / "split_indices.npz", context=ctx_idx, test=test_idx)

    with open(output_dir / "info.pkl", "wb") as f:
        pickle.dump({
            "dataset": "gamma_zero", "n_classes": n_cls,
            "class_names": CLASS_NAMES,
            "n_context": len(ctx_idx), "n_test": len(test_idx),
            "modalities": ["fundus", "oct"],
            "encoder": "resnet18 (fundus) + resnet18 (OCT avg)",
            "normalization": "none; raw encoder outputs retained without dataset-level normalization",
            "normalization_fit_scope": "none",
        }, f)
    print(f"Done: {output_dir}")


if __name__ == "__main__":
    main()

