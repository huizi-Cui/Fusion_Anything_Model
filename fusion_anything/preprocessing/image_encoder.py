"""Frozen image encoders and split helper used by the Derm7pt example."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def build_resnet18_encoder(device, weights_path=None):
    """Build a frozen ResNet18 encoder returning pooled image features.
    
    A supplied weights_path uses a local checkpoint. Otherwise torchvision
    loads its default pretrained ImageNet weights and may download them."""
    import torchvision.models as models
    backbone = models.resnet18(weights=None if weights_path else "DEFAULT")
    if weights_path:
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        missing = [key for key in missing if not key.endswith("num_batches_tracked")]
        if missing or unexpected:
            raise RuntimeError(
                f"Invalid ResNet-18 checkpoint {weights_path}: "
                f"missing={missing}, unexpected={unexpected}"
            )
    encoder = nn.Sequential(
        backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
        backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        nn.AdaptiveAvgPool2d(1),
    )
    encoder = encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def extract_resnet18_features(encoder, image_paths, device, batch_size=64, num_workers=8):
    """Encode images in their input order as pooled 512-dimensional features.
    
    Images are resized and normalized using ImageNet statistics. The existing
    adapter substitutes a zero image tensor if decoding fails; callers needing
    strict image validation should validate their paths before extraction."""
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import Compose, Normalize, Resize, ToTensor

    transform = Compose([Resize((224, 224)), ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    class _ImagePathDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = [str(path) for path in paths]
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, index):
            try:
                image = Image.open(self.paths[index]).convert("RGB")
                return self.transform(image)
            except Exception:
                return torch.zeros(3, 224, 224)

    loader = DataLoader(_ImagePathDataset(image_paths, transform), batch_size=batch_size,
                        num_workers=num_workers, shuffle=False, pin_memory=True)
    features = []
    with torch.no_grad():
        for batch in loader:
            output = encoder(batch.to(device))
            features.append(output.flatten(1).cpu().numpy())
    return np.concatenate(features, axis=0)


def split_without_normalization(X, ctx_idx, test_idx):
    """Select Context/Test rows as independent float32 copies, without normalization."""
    matrix = np.asarray(X)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"X must be a non-empty 2D matrix, got {matrix.shape}")
    context = np.asarray(matrix[np.asarray(ctx_idx)], dtype=np.float32).copy()
    test = np.asarray(matrix[np.asarray(test_idx)], dtype=np.float32).copy()
    return context, test
