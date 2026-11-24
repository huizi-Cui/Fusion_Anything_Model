"""Frozen DINOv3 and Qwen3 helpers for the MultiOFF example.

Numerical transforms follow the maintained MultiOFF feature pipeline.
Model source/weight locations are explicit; no developer machine paths are used.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageFile

def _load_rgb_allow_truncated(path):
    try:
        with Image.open(str(path)) as image:
            return image.convert("RGB")
    except OSError as error:
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            with Image.open(str(path)) as image:
                rgb = image.convert("RGB")
                rgb.load()
            print(f"  WARNING: tolerated truncated image: {path} ({error})", flush=True)
            return rgb
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous


def build_dinov3_vitb16_encoder(device, repository: Path, weights: Path):
    """Load a frozen DINOv3 ViT-B/16 from explicit local source and weights."""
    import sys

    repository = Path(repository).expanduser().resolve()
    weights = Path(weights).expanduser().resolve()
    if not repository.is_dir():
        raise FileNotFoundError(f"DINOv3 repository not found: {repository}")
    if not weights.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights}")
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from dinov3.hub.backbones import dinov3_vitb16

    model = dinov3_vitb16(pretrained=False)
    model.load_state_dict(
        torch.load(weights, map_location="cpu", weights_only=True), strict=True
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device).eval()


def extract_dinov3_features(model, image_paths, device, batch_size=32):
    """Encode images at 256x256 with ImageNet statistics and BF16 CUDA autocast."""
    from torchvision.transforms import v2

    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((256, 256), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    features = []
    with torch.inference_mode():
        for start in range(0, len(image_paths), batch_size):
            images = [
                transform(_load_rgb_allow_truncated(path))
                for path in image_paths[start:start + batch_size]
            ]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(torch.stack(images).to(device))
            features.append(output.float().cpu().numpy())
    return np.concatenate(features, axis=0).astype(np.float32)


def _last_token_pool(last_hidden_state, attention_mask):
    if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        sequence_lengths,
    ]


def extract_qwen3_embedding_features(
    texts,
    device,
    model_path,
    batch_size=16,
    max_length=512,
):
    """Encode text using last-token pooling and per-row L2 normalization."""
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    model_path = str(model_path)
    local_only = Path(model_path).is_dir()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_only,
        padding_side="left",
    )
    model_kwargs = {
        "local_files_only": local_only,
    }
    if device.type == "cuda":
        model_kwargs["dtype"] = torch.bfloat16
    model = AutoModel.from_pretrained(model_path, **model_kwargs).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    features = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = [str(text) if text else "" for text in texts[start:start + batch_size]]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            output = model(**encoded)
            embedding = _last_token_pool(output.last_hidden_state, encoded["attention_mask"])
            features.append(functional.normalize(embedding.float(), p=2, dim=1).cpu().numpy())
    return np.concatenate(features, axis=0).astype(np.float32)
