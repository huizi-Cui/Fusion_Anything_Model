"""Frozen feature encoders. Local model paths are defined in dataset_paths.py."""

from __future__ import annotations
import os
from pathlib import Path
from .dataset_paths import encoder_path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile


ENCODER_ROOT = Path(
    os.environ.get(
        "FUSION_ENCODER_ROOT",
        str(encoder_path("")),
    )
)
HF_HUB_CACHE = ENCODER_ROOT / "huggingface/hub"
DINOV3_MODEL_ROOT = Path(
    os.environ.get(
        "FUSION_DINOV3_ROOT",
        str(encoder_path("dinov3")),
    )
)


# DINOv2 ViT-S/14 — 384d


def build_dinov2_encoder(device):
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def extract_dinov2_features(model, image_paths, device, batch_size=64, num_workers=8):
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import Compose, Normalize, Resize, ToTensor

    transform = Compose([
        Resize((224, 224)), ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class _ImagePathDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = [str(p) for p in paths]
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            try:
                img = Image.open(self.paths[idx]).convert("RGB")
                return self.transform(img)
            except Exception:
                return torch.zeros(3, 224, 224)

    dataset = _ImagePathDataset(image_paths, transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, pin_memory=True)
    feats = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch.to(device))  # [B, 384]
            feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0)


# ResNet-18 ImageNet — 512d (用于深度图/RGB-D)


def build_resnet18_encoder(device, weights_path=None):
    import torchvision.models as models
    if weights_path is None:
        weights_path = encoder_path("resnet18/resnet18-f37072fd.pth")
    backbone = models.resnet18(weights=None if weights_path else "DEFAULT")
    if weights_path:
        # torchvision's legacy ``resnet18-5c106cde.pth`` uses the old tar
        # serialization format, which is incompatible with weights_only=True.
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
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def extract_resnet18_features(encoder, image_paths, device, batch_size=64, num_workers=8):
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import Compose, Normalize, Resize, ToTensor

    transform = Compose([Resize((224, 224)), ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    class _ImagePathDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = [str(p) for p in paths]
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            try:
                img = Image.open(self.paths[idx]).convert("RGB")
                return self.transform(img)
            except Exception:
                return torch.zeros(3, 224, 224)

    dataset = _ImagePathDataset(image_paths, transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, pin_memory=True)
    feats = []
    with torch.no_grad():
        for batch in loader:
            out = encoder(batch.to(device))
            feats.append(out.flatten(1).cpu().numpy())
    return np.concatenate(feats, axis=0)

BERT_PATH = os.environ.get("FUSION_BERT_PATH", str(encoder_path("bert")))


def extract_bert_features(texts, device, batch_size=64):
    """BERT mean pooling → [N, 768], 走 MGM bucket 自动处理维度"""
    from transformers import BertModel, BertTokenizer
    model = BertModel.from_pretrained(BERT_PATH).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    feats = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [str(t) if t else "" for t in texts[i:i + batch_size]]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=128, return_tensors="pt").to(device)
            out = model(**enc)
            # Mean pooling over all tokens (excluding [CLS]-only bias)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            feats.append(emb.cpu().numpy())
    return np.concatenate(feats, axis=0)


# MiniLM mean pooling — 384d


MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def extract_minilm_features(texts, device, batch_size=64):
    """MiniLM mean pooling → [N, 384], 维度 ≤512，MGM 可直接处理."""
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(MINILM_NAME).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(MINILM_NAME)
    feats = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [str(t) if t else "" for t in texts[i:i + batch_size]]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=128, return_tensors="pt").to(device)
            out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            feats.append(emb.cpu().numpy())
    return np.concatenate(feats, axis=0)


# DINOv2 ViT-B/14 — 768d


DINOV2B_NAME = "facebook/dinov2-base"


def build_dinov2_vitb14_encoder(device):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(DINOV2B_NAME).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def extract_dinov2_vitb14_features(model, image_paths, device, batch_size=32, num_workers=8):
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import Compose, Normalize, Resize, ToTensor

    transform = Compose([
        Resize((224, 224)), ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class _ImagePathDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = [str(p) for p in paths]
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            try:
                img = Image.open(self.paths[idx]).convert("RGB")
                return self.transform(img)
            except Exception:
                return torch.zeros(3, 224, 224)

    dataset = _ImagePathDataset(image_paths, transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, pin_memory=True)
    feats = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch.to(device))
            out = out.last_hidden_state[:, 0, :]  # [CLS], [B, 768]
            feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0)


# Depth Anything V2 Base — 768d (DINOv2 ViT-B backbone, 合成数据训练, 未见过NYUDv2)


DEPTH_ANYTHING_V2_BASE = "/path/to/encoder_weights/Depth-Anything-V2-Base-hf"


def build_depthanythingv2_encoder(device):
    '加载 Depth Anything V2 Base 的 DINOv2 ViT-B backbone (仅编码器, 冻结)。.'
    from transformers import AutoModelForDepthEstimation
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_ANYTHING_V2_BASE)
    # 只用 backbone (DINOv2 ViT), 丢弃 neck 和 head
    encoder = model.backbone
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def extract_depthanythingv2_features(encoder, image_paths, device, batch_size=32):
    '用 Depth Anything V2 backbone 提取深度图 CLS token 特征 → [N, 768]。.'
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize
    transform = Compose([
        Resize((518, 518)), ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    feats = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch = []
            for p in image_paths[i:i + batch_size]:
                try:
                    img = Image.open(str(p)).convert("RGB")
                    batch.append(transform(img))
                except Exception:
                    batch.append(torch.zeros(3, 518, 518))
            out = encoder(torch.stack(batch).to(device))
            # feature_maps: tuple of 4 tensors [B, N, 768]
            # 取最后一层的 CLS token → [B, 768]
            cls_feat = out.feature_maps[-1][:, 0, :]
            feats.append(cls_feat.cpu().numpy())
    return np.concatenate(feats, axis=0)


# Depth Pro (Apple 2024) — 1024d (DINOv2 ViT-L backbone, 多分辨率深度预训练)


DEPTH_PRO_HF = "/path/to/encoder_weights/DepthPro-hf"


def build_depthpro_encoder(device):
    '加载 Depth Pro 的 image_encoder (DINOv2 ViT-L backbone, 仅编码器, 冻结)。.'
    from transformers import AutoModelForDepthEstimation
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_PRO_HF)
    # 使用 image_encoder (DINOv2 ViT-L), 丢弃 patch_encoder/neck/head/fov
    encoder = model.depth_pro.encoder.image_encoder.model
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def extract_depthpro_features(encoder, image_paths, device, batch_size=16):
    '用 Depth Pro image_encoder (DINOv2 ViT-L) 提取 CLS token → [N, 1024]。.'
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize
    transform = Compose([
        Resize((384, 384)), ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    feats = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch = []
            for p in image_paths[i:i + batch_size]:
                try:
                    img = Image.open(str(p)).convert("RGB")
                    batch.append(transform(img))
                except Exception:
                    batch.append(torch.zeros(3, 384, 384))
            out = encoder(torch.stack(batch).to(device))
            cls_feat = out.last_hidden_state[:, 0, :]  # [B, 1024]
            feats.append(cls_feat.cpu().numpy())
    return np.concatenate(feats, axis=0)


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


def build_dinov3_vitb16_encoder(device):
    """Load frozen DINOv3-B checkpoint without changing its treatment."""
    import sys

    repository = DINOV3_MODEL_ROOT / "dinov3_repo"
    weights = DINOV3_MODEL_ROOT / "dinov3_vit_b/backbone.pth"
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
    return model.to(device).eval()


def extract_dinov3_features(model, image_paths, device, batch_size=32):
    """Reproduce 256x256 DINOv3 extraction and BF16 CUDA autocast."""
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


QWEN3_EMBEDDING_NAME = "Qwen/Qwen3-Embedding-0.6B"


def _resolve_hf_snapshot(repo_id: str) -> str:
    repository = HF_HUB_CACHE / ("models--" + repo_id.replace("/", "--"))
    ref = repository / "refs/main"
    snapshots = repository / "snapshots"
    if ref.is_file():
        candidate = snapshots / ref.read_text(encoding="utf-8").strip()
        if candidate.is_dir():
            return str(candidate)
    if snapshots.is_dir():
        candidates = sorted(path for path in snapshots.iterdir() if path.is_dir())
        if candidates:
            return str(candidates[-1])
    return repo_id


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
    batch_size=16,
    max_length=512,
):
    """Reproduce last-token pooled, L2-normalized Qwen3 embeddings."""
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    model_path = str(encoder_path("qwen3"))
    local_only = os.path.isdir(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=str(HF_HUB_CACHE),
        local_files_only=local_only,
        padding_side="left",
    )
    model_kwargs = {
        "cache_dir": str(HF_HUB_CACHE),
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


# ModernBERT-base — 768d


MODERNBERT_NAME = "answerdotai/ModernBERT-base"


def extract_modernbert_features(texts, device, batch_size=64):
    from transformers import AutoTokenizer, AutoModel
    model = AutoModel.from_pretrained(MODERNBERT_NAME).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(MODERNBERT_NAME)
    feats = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [str(t) if t else "" for t in texts[i:i + batch_size]]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=512, return_tensors="pt").to(device)
            out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            feats.append(emb.cpu().numpy())
    return np.concatenate(feats, axis=0)


def quantile_normalize(X: np.ndarray) -> np.ndarray:
    'Split-wise average-rank Gaussian transform.'
    from scipy.stats import norm as scipy_norm
    from scipy.stats import rankdata

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 1:
        raise ValueError("normalization requires a non-empty 2D array")
    if not np.isfinite(X).all():
        raise ValueError("normalization input contains NaN or infinite values")

    n = X.shape[0]
    ranks = rankdata(X, method="average", axis=0, nan_policy="raise")
    quantiles = ranks / (n + 2)
    return scipy_norm.ppf(np.clip(quantiles, 1e-8, 1 - 1e-8)).astype(np.float32)


def quantile_normalize_splitwise(
    X: np.ndarray,
    ctx_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    '分别在 Context 和完整 Test 上拟合 average-rank 秩高斯变换。.'
    matrix = np.asarray(X)
    context = matrix[np.asarray(ctx_idx)]
    test = matrix[np.asarray(test_idx)]
    return quantile_normalize(context), quantile_normalize(test)


class ContextQuantileGaussianTransformer:
    """仅在 Context 上拟合的 tie-aware 经验分位高斯变换。"""

    def __init__(self, output_dtype=np.float32):
        self.output_dtype = output_dtype
        self.n_context_ = None
        self.n_features_in_ = None
        self.sorted_context_ = None

    @staticmethod
    def _finite_matrix(values: np.ndarray, name: str) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError(f"{name} must be a non-empty 2D matrix, got {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} contains NaN or infinite values")
        return matrix

    def fit(self, context: np.ndarray):
        matrix = self._finite_matrix(context, "context")
        self.n_context_, self.n_features_in_ = map(int, matrix.shape)
        self.sorted_context_ = tuple(
            np.sort(matrix[:, feature], kind="mergesort")
            for feature in range(self.n_features_in_)
        )
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        from scipy.special import ndtri

        if self.sorted_context_ is None:
            raise RuntimeError("ContextQuantileGaussianTransformer is not fitted")
        matrix = self._finite_matrix(values, "values")
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature mismatch: fitted on {self.n_features_in_}, got {matrix.shape[1]}"
            )

        transformed = np.empty(matrix.shape, dtype=np.float64)
        lower_probability = 0.5 / self.n_context_
        upper_probability = 1.0 - lower_probability
        for feature, reference in enumerate(self.sorted_context_):
            if reference[0] == reference[-1]:
                transformed[:, feature] = 0.0
                continue
            column = matrix[:, feature]
            left = np.searchsorted(reference, column, side="left")
            right = np.searchsorted(reference, column, side="right")
            probabilities = np.where(
                right > left,
                (left + right) / (2.0 * self.n_context_),
                left / float(self.n_context_),
            )
            transformed[:, feature] = ndtri(
                np.clip(probabilities, lower_probability, upper_probability)
            )
        return transformed.astype(self.output_dtype, copy=False)

    def fit_transform(self, context: np.ndarray) -> np.ndarray:
        return self.fit(context).transform(context)


def quantile_normalize_split(X: np.ndarray, ctx_idx: np.ndarray, test_idx: np.ndarray):
    '历史接口：仅用 Context 拟合 tie-aware 秩高斯变换，再复用于 Test。.'
    context = np.asarray(X)[ctx_idx]
    test = np.asarray(X)[test_idx]
    transformer = ContextQuantileGaussianTransformer().fit(context)
    return transformer.transform(context), transformer.transform(test)


class ContextZScoreTransformer:
    'TabPFN-style z-score fitted only on Context rows.'

    def __init__(self, output_dtype=np.float32):
        self.output_dtype = np.dtype(output_dtype)
        if not np.issubdtype(self.output_dtype, np.floating):
            raise TypeError(f"output_dtype must be floating, got {self.output_dtype}")
        self.n_features_in_ = None
        self.mean_ = None
        self.scale_ = None

    @staticmethod
    def _matrix(values: np.ndarray, name: str) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError(f"{name} must be a non-empty 2D matrix, got {matrix.shape}")
        if np.isinf(matrix).any():
            raise ValueError(f"{name} contains infinite values")
        return matrix

    def fit(self, context: np.ndarray):
        matrix = self._matrix(context, "context")
        self.n_features_in_ = int(matrix.shape[1])

        nan_mask = np.isnan(matrix)
        valid_count = (~nan_mask).sum(axis=0)
        value_sum = np.where(nan_mask, 0.0, matrix).sum(axis=0)
        mean = value_sum / np.maximum(valid_count, 1)
        squared_difference = np.where(nan_mask, 0.0, (matrix - mean) ** 2).sum(axis=0)
        variance = squared_difference / np.maximum(valid_count - 1, 1)
        scale = np.sqrt(variance)
        scale = np.where((valid_count <= 1) | (scale == 0), 1.0, scale)

        self.mean_ = mean.astype(self.output_dtype, copy=False)
        self.scale_ = scale.astype(self.output_dtype, copy=False)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("ContextZScoreTransformer is not fitted")
        matrix = self._matrix(values, "values")
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature mismatch: fitted on {self.n_features_in_}, got {matrix.shape[1]}"
            )
        matrix = matrix.astype(self.output_dtype, copy=False)
        epsilon = np.finfo(self.output_dtype).eps
        transformed = (matrix - self.mean_) / (self.scale_ + epsilon)
        return np.clip(transformed, -100.0, 100.0).astype(self.output_dtype, copy=False)

    def fit_transform(self, context: np.ndarray) -> np.ndarray:
        return self.fit(context).transform(context)


def zscore_normalize_split(
    X: np.ndarray,
    ctx_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    'Fit TabPFN-style z-score on Context and reuse it for Test.'
    matrix = np.asarray(X)
    context = matrix[np.asarray(ctx_idx)]
    test = matrix[np.asarray(test_idx)]
    transformer = ContextZScoreTransformer(output_dtype=np.float32).fit(context)
    return transformer.transform(context), transformer.transform(test)


def split_without_normalization(
    X: np.ndarray,
    ctx_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    'Select Context and Test rows without dataset-level normalization.'
    matrix = np.asarray(X)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"X must be a non-empty 2D matrix, got {matrix.shape}")
    context = np.asarray(matrix[np.asarray(ctx_idx)], dtype=np.float32).copy()
    test = np.asarray(matrix[np.asarray(test_idx)], dtype=np.float32).copy()
    return context, test


# SequenceTransformer — 512d (随机初始化, 冻结, 用于时序数据如MIMIC-III)


def build_seq_transformer_encoder(device, input_dim=12, hidden_dim=512, num_heads=4, num_layers=1):
    """构建冻结的 SequenceTransformerEncoder。随机初始化，零训练。"""
    from encoders.sequence_transformer import SequenceTransformerEncoder
    model = SequenceTransformerEncoder(input_dim, hidden_dim, num_heads, num_layers)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def extract_seq_transformer_features(model, sequences_3d, device, batch_size=64):
    """从 3D 时序数据 (N, T, D) 提取特征 → [N, hidden_dim]"""
    feats = []
    with torch.no_grad():
        for i in range(0, len(sequences_3d), batch_size):
            batch = sequences_3d[i:i + batch_size]
            x = torch.from_numpy(batch).float().to(device)
            out = model(x)  # [B, hidden_dim]
            feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0)


# SequenceLSTMEncoder — LSTM 时序编码器 (用于 EEG 等时序数据)


def build_sequence_lstm_encoder(device, input_dim=192, hidden_dim=512, num_layers=2):
    """构建冻结的 SequenceLSTMEncoder。随机初始化，零训练。"""
    from encoders.sequence_lstm import SequenceLSTMEncoder
    model = SequenceLSTMEncoder(input_dim, hidden_dim, num_layers)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def extract_sequence_lstm_features(model, sequences_3d, device, batch_size=64):
    """从 3D 时序数据 (N, T, D) 提取 LSTM 特征 → [N, hidden_dim]"""
    feats = []
    with torch.no_grad():
        for i in range(0, len(sequences_3d), batch_size):
            batch = sequences_3d[i:i + batch_size]
            x = torch.from_numpy(batch).float().to(device)
            out = model(x)  # [B, hidden_dim]
            feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0)

