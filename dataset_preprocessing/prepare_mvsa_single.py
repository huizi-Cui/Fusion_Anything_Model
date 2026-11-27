"""MVSA-Single images and text encoded with ResNet-18 and BERT."""

from __future__ import annotations
import pickle, sys, json
from pathlib import Path
from .dataset_paths import dataset_path
import numpy as np; import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from .encoders_zero import (
    build_resnet18_encoder, extract_resnet18_features,
    extract_bert_features, split_without_normalization,
)

DATA_DIR = dataset_path('datasets/23_MVSA-Single/data/MVSA_Single')
LABEL_MAP = {"positive": 0, "neutral": 1, "negative": 2}
CLASS_NAMES = ["positive", "neutral", "negative"]

def load_jsonl(path):
    samples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))
    return samples


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = ROOT / "data/mvsa_single_zero"
    output_dir.mkdir(parents=True, exist_ok=True)


    train_d = load_jsonl(DATA_DIR / "train.jsonl")
    val_d = load_jsonl(DATA_DIR / "val.jsonl")
    test_d = load_jsonl(DATA_DIR / "test.jsonl")
    all_d = train_d + val_d + test_d
    n_tr, n_va = len(train_d), len(val_d)

    labels = np.array([LABEL_MAP[d["label"]] for d in all_d], dtype=np.int64)
    image_paths = [str(DATA_DIR / d["img"]) for d in all_d]
    texts = [d["text"] for d in all_d]
    ctx_idx = np.arange(0, n_tr + n_va)
    test_idx = np.arange(n_tr + n_va, len(all_d))
    print(f"MVSA-Single: {len(all_d)} samples, 3 classes, ctx={len(ctx_idx)}, test={len(test_idx)}")


    print("[Image] encoder=ImageNet ResNet-18")
    img_dim = 512
    img_encoder = build_resnet18_encoder(device)
    img_raw = extract_resnet18_features(img_encoder, image_paths, device)
    del img_encoder
    torch.cuda.empty_cache()
    print(f"  Image: {img_raw.shape} ({img_dim}d)")


    print("[Text] encoder=BERT")
    txt_dim = 768
    txt_raw = extract_bert_features(texts, device)
    print(f"  Text: {txt_raw.shape} ({txt_dim}d)")


    print("  Saving raw Context/Test feature splits without dataset-level normalization...")
    ctx_img, test_img = split_without_normalization(img_raw, ctx_idx, test_idx)
    ctx_txt, test_txt = split_without_normalization(txt_raw, ctx_idx, test_idx)

    np.save(output_dir / "context_image.npy", ctx_img)
    np.save(output_dir / "test_image.npy", test_img)
    np.save(output_dir / "context_text.npy", ctx_txt)
    np.save(output_dir / "test_text.npy", test_txt)
    np.save(output_dir / "context_labels.npy", labels[ctx_idx])
    np.save(output_dir / "test_labels.npy", labels[test_idx])
    np.savez(output_dir / "split_indices.npz", context=ctx_idx, test=test_idx)

    info = {
        "dataset": "mvsa_single_zero", "n_classes": 3, "class_names": CLASS_NAMES,
        "n_context": len(ctx_idx), "n_test": len(test_idx),
        "encoder": "ImageNet ResNet-18 (512d) + BERT (768d)",
        "image_dim": img_dim, "text_dim": txt_dim,
        "normalization": "none; raw encoder outputs retained without dataset-level normalization",
        "normalization_fit_scope": "none",
    }
    with open(output_dir / "info.pkl", "wb") as f:
        pickle.dump(info, f)
    print(f"Done: {output_dir}")


if __name__ == "__main__":
    main()

