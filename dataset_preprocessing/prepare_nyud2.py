"""NYUDv2 RGB and depth images encoded with ImageNet ResNet-18."""

from __future__ import annotations
import pickle, sys
from pathlib import Path
from .dataset_paths import dataset_path
import numpy as np; import torch
from PIL import Image
from torchvision.datasets.folder import make_dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from .encoders_zero import build_resnet18_encoder, split_without_normalization

DATA_DIR = dataset_path('datasets/03_NYUDv2/data/nyud2_trainvaltest')
IMAGE_SIZE, BATCH_SIZE = 224, 64


def load_split(split_dir):
    entries = sorted([d for d in split_dir.iterdir() if d.is_dir()], key=lambda d: d.name)
    class_to_idx = {d.name: i for i, d in enumerate(entries)}
    imgs = make_dataset(str(split_dir), class_to_idx, ('png', 'jpg', 'jpeg'))
    return [Path(p) for p, _ in imgs], np.array([l for _, l in imgs], dtype=np.int64), [d.name for d in entries]


def split_rgb_depth(img):
    w, h = img.size; w2 = w // 2
    return img.crop((0, 0, w2, h)), img.crop((w2, 0, w, h))


def extract_modality(encoder, paths, device, is_depth):
    """使用普通 ImageNet ResNet-18 提取 RGB 或 Depth 半图的特征。"""
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize
    transform = Compose([
        Resize((IMAGE_SIZE, IMAGE_SIZE)), ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), BATCH_SIZE):
            batch = []
            for path in paths[i:i + BATCH_SIZE]:
                try:
                    full = Image.open(str(path)).convert("RGB")
                    rgb, depth = split_rgb_depth(full)
                    batch.append(transform(depth if is_depth else rgb))
                except Exception:
                    batch.append(torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE))
            output = encoder(torch.stack(batch).to(device))
            feats.append(output.flatten(1).cpu().numpy())
    return np.concatenate(feats, axis=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = ROOT / "data/nyud2_zero"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_p, train_l, classes = load_split(DATA_DIR / "train")
    val_p, val_l, _ = load_split(DATA_DIR / "val")
    test_p, test_l, _ = load_split(DATA_DIR / "test")

    # This data package stores validation as a subset of train.  Do not append
    # val again: doing so duplicates all 414 validation images in context.
    def sample_key(path):
        return (path.parent.name, path.name)

    train_keys = {sample_key(path): label for path, label in zip(train_p, train_l)}
    val_keys = {sample_key(path): label for path, label in zip(val_p, val_l)}
    test_keys = {sample_key(path): label for path, label in zip(test_p, test_l)}
    if not val_keys.keys() <= train_keys.keys():
        missing = len(val_keys.keys() - train_keys.keys())
        raise ValueError(f"NYUD2 validation is not a train subset: {missing} missing")
    if any(train_keys[key] != label for key, label in val_keys.items()):
        raise ValueError("NYUD2 train/validation labels disagree")
    if train_keys.keys() & test_keys.keys():
        raise ValueError("NYUD2 train and test contain overlapping images")

    n_train, n_val, n_test = len(train_p), len(val_p), len(test_p)
    n_train_only = n_train - n_val
    all_p = np.concatenate([train_p, test_p])
    all_l = np.concatenate([train_l, test_l])
    n_cls = len(classes)
    print(f"NYUD2 zero (official split): {n_cls} classes, {len(all_p)} unique total")
    print(f"  Fixed split: train-only={n_train_only}, validation={n_val}, "
          f"context={n_train}, test={n_test}")

    ctx_idx = np.arange(0, n_train)
    test_idx = np.arange(n_train, len(all_p))
    print(f"  Context: {len(ctx_idx)}, Test: {len(test_idx)}")

    print("[RGB] encoder=resnet18")
    rgb_encoder = build_resnet18_encoder(device)
    rgb_raw = extract_modality(rgb_encoder, all_p, device, is_depth=False)
    del rgb_encoder
    torch.cuda.empty_cache()

    print("[Depth] encoder=resnet18")
    depth_encoder = build_resnet18_encoder(device)
    depth_raw = extract_modality(depth_encoder, all_p, device, is_depth=True)
    del depth_encoder
    torch.cuda.empty_cache()

    print(f"  RGB: {rgb_raw.shape}, Depth: {depth_raw.shape}")

    context_rgb, test_rgb = split_without_normalization(rgb_raw, ctx_idx, test_idx)
    context_depth, test_depth = split_without_normalization(depth_raw, ctx_idx, test_idx)
    np.save(output_dir / "context_rgb.npy", context_rgb)
    np.save(output_dir / "test_rgb.npy", test_rgb)
    np.save(output_dir / "context_depth.npy", context_depth)
    np.save(output_dir / "test_depth.npy", test_depth)
    np.save(output_dir / "context_labels.npy", all_l[ctx_idx])
    np.save(output_dir / "test_labels.npy", all_l[test_idx])
    np.savez(output_dir / "split_indices.npz", context=ctx_idx, test=test_idx)

    with open(output_dir / "info.pkl", "wb") as f:
        pickle.dump({"dataset": "nyud2_zero", "n_classes": n_cls, "class_names": classes,
                      "n_context": len(ctx_idx), "n_test": len(test_idx),
                      "n_train_only": n_train_only, "n_validation": n_val,
                      "split_seed": None,
                      "split": "fixed directories; val is a subset of train; context=train without duplication; test=test",
                      "encoder": "rgb=resnet18, depth=resnet18",
                      "normalization": "none; raw encoder outputs retained without dataset-level normalization",
                      "normalization_fit_scope": "none"}, f)
    print(f"Done: {output_dir}")


if __name__ == "__main__":
    main()

