"""PAD-UFES-20 uses ResNet-50 and structured metadata with five patient-grouped folds."""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path
from .dataset_paths import dataset_path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from .migration_common import save_dataset

RAW_DIR = dataset_path('datasets/17_PAD-UFES-20/data/pad_ufes_20')
ARCHIVE = dataset_path('datasets/17_PAD-UFES-20/data/pad_ufes_20/pad_ufes_20_complete_kaggle.zip')
METADATA = dataset_path('datasets/17_PAD-UFES-20/data/pad_ufes_20/metadata.csv')
CACHE = dataset_path('datasets/17_PAD-UFES-20/data/padufes20_resnet50.npy')
OUTPUT_ROOT = ROOT / "data"
CLASSES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]


def metadata_features(frame, context_idx):
    excluded = {"patient_id", "lesion_id", "img_id", "diagnostic", "biopsed"}
    numeric = ["age", "diameter_1", "diameter_2"]
    categorical = ["background_father", "background_mother", "gender", "fitspatrick", "region"]
    boolean = [x for x in frame.columns if x not in excluded and x not in numeric and x not in categorical]
    context = frame.iloc[context_idx]
    columns = []
    for name in numeric:
        raw = pd.to_numeric(frame[name], errors="coerce")
        median = float(pd.to_numeric(context[name], errors="coerce").median())
        columns += [raw.fillna(median).to_numpy(np.float32)[:, None], raw.isna().to_numpy(np.float32)[:, None]]
    for name in boolean:
        mapped = frame[name].map({True: 1., False: 0., "True": 1., "False": 0.})
        columns += [mapped.fillna(0.).to_numpy(np.float32)[:, None], mapped.isna().to_numpy(np.float32)[:, None]]
    for name in categorical:
        context_values = context[name].fillna("__MISSING__").astype(str)
        categories = sorted(set(context_values))
        values = frame[name].fillna("__MISSING__").astype(str)
        columns += [(values == category).to_numpy(np.float32)[:, None] for category in categories]
        columns.append((~values.isin(categories)).to_numpy(np.float32)[:, None])
    return np.concatenate(columns, axis=1).astype(np.float32)


@torch.inference_mode()
def encode_images(frame, device, batch_size):
    from torchvision.models import ResNet50_Weights, resnet50
    with zipfile.ZipFile(ARCHIVE) as handle:
        by_name = {Path(x).name: x for x in handle.namelist()
                   if x.lower().endswith((".png", ".jpg", ".jpeg")) and not Path(x).name.startswith("._")}
        members = [by_name[str(x)] for x in frame.img_id]
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights); model.fc = torch.nn.Identity()
        model = model.to(device).eval().requires_grad_(False)
        output = []
        for start in range(0, len(members), batch_size):
            images = [weights.transforms()(Image.open(io.BytesIO(handle.read(x))).convert("RGB"))
                      for x in members[start:start + batch_size]]
            output.append(F.normalize(model(torch.stack(images).to(device)).float(), dim=1).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--image-cache", type=Path, default=CACHE)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--reextract", action="store_true")
    parser.add_argument("--pca48", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(METADATA)
    label_map = {x: i for i, x in enumerate(CLASSES)}
    labels = frame.diagnostic.map(label_map).to_numpy(np.int64)
    if args.reextract or not args.image_cache.is_file():
        image = encode_images(frame, torch.device(args.device), args.image_batch_size)
        args.image_cache.parent.mkdir(parents=True, exist_ok=True); np.save(args.image_cache, image)
    else:
        image = np.load(args.image_cache)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (context, test) in enumerate(folds.split(frame, labels, groups=frame.patient_id)):
        context = np.asarray(context, dtype=np.int64); test = np.asarray(test, dtype=np.int64)
        metadata = metadata_features(frame, context)
        output = args.output_root / f"padufes20_patient_sgkf{fold}_resnet50_meta"
        save_dataset(output, labels, context, test, {"image": image, "metadata": metadata}, {
            "display_name": f"PAD-UFES-20 fold {fold}", "n_classes": 6, "class_names": CLASSES,
            "label_map": label_map, "modalities": ["image", "metadata"], "structured_modalities": ["metadata"],
            "image_encoder": "torchvision ResNet-50 ImageNet1K V2; L2-normalized", "image_dim": 2048,
            "metadata_policy": "Context median+missing flags; boolean 0/1+missing; Context one-hot+unknown",
            "excluded_metadata": ["patient_id", "lesion_id", "img_id", "diagnostic", "biopsed"],
            "split": f"StratifiedGroupKFold fold {fold}; patient_id groups; shuffle seed 42",
            "context_patients": int(frame.iloc[context].patient_id.nunique()),
            "test_patients": int(frame.iloc[test].patient_id.nunique()),
            "patient_overlap": len(set(frame.iloc[context].patient_id) & set(frame.iloc[test].patient_id)),
            "source": str(ARCHIVE),
        }, {"context": context, "test": test}, args.pca48)


if __name__ == "__main__":
    main()

