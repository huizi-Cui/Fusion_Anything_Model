"""Shared image and text feature extraction."""

from __future__ import annotations

import pickle
import re
import unicodedata
from pathlib import Path

import numpy as np
import torch

from .encoders_zero import (
    build_dinov3_vitb16_encoder,
    build_resnet18_encoder,
    extract_dinov3_features,
    extract_bert_features,
    extract_qwen3_embedding_features,
    extract_resnet18_features,
    split_without_normalization,
)


def clean_text(value) -> str:
    """Match control-character and whitespace cleanup exactly."""
    text = str(value) if value is not None else ""
    text = "".join(
        character
        for character in text
        if character in "\n\t" or unicodedata.category(character) != "Cc"
    )
    text = re.sub(r"[^\S\n]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def label_distribution(labels: np.ndarray, class_names: list[str]) -> dict[str, int]:
    return {
        class_names[class_id]: int(np.count_nonzero(labels == class_id))
        for class_id in range(len(class_names))
    }


def encode_save_without_normalization(
    *,
    output_dir: Path,
    device: torch.device,
    image_paths: list[str],
    texts: list[str],
    labels: np.ndarray,
    context_indices: np.ndarray,
    test_indices: np.ndarray,
    split_indices: dict[str, np.ndarray],
    info: dict,
    image_encoder: str = "dinov3_b",
    text_encoder: str = "qwen3_embedding_0_6b",
) -> Path:
    """Apply the preserved encoders and save their raw outputs without dataset-level normalization."""
    missing = [path for path in image_paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first={missing[0]}")

    if image_encoder == "dinov3_b":
        print("Extracting image features: DINOv3 ViT-B/16", flush=True)
        image_model = build_dinov3_vitb16_encoder(device)
        image_features = extract_dinov3_features(image_model, image_paths, device)
        image_dim = 768
        image_description = "DINOv3 ViT-B/16"
    elif image_encoder == "resnet18":
        print("Extracting image features: ImageNet ResNet-18", flush=True)
        image_model = build_resnet18_encoder(device)
        image_features = extract_resnet18_features(image_model, image_paths, device)
        image_dim = 512
        image_description = "ImageNet ResNet-18"
    else:
        raise ValueError(f"Unsupported image encoder: {image_encoder}")
    del image_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if text_encoder == "qwen3_embedding_0_6b":
        print("Extracting text features: Qwen3-Embedding-0.6B", flush=True)
        text_features = extract_qwen3_embedding_features(texts, device)
        text_dim = 1024
        text_description = "Qwen3-Embedding-0.6B"
    elif text_encoder == "bert":
        print("Extracting text features: BERT-base-uncased CLS", flush=True)
        text_features = extract_bert_features(texts, device)
        text_dim = 768
        text_description = "BERT-base-uncased CLS"
    else:
        raise ValueError(f"Unsupported text encoder: {text_encoder}")
    context_image, test_image = split_without_normalization(
        image_features, context_indices, test_indices
    )
    context_text, test_text = split_without_normalization(
        text_features, context_indices, test_indices
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "context_image.npy", context_image)
    np.save(output_dir / "test_image.npy", test_image)
    np.save(output_dir / "context_text.npy", context_text)
    np.save(output_dir / "test_text.npy", test_text)
    np.save(output_dir / "context_labels.npy", labels[context_indices])
    np.save(output_dir / "test_labels.npy", labels[test_indices])
    np.save(output_dir / "context_indices.npy", context_indices)
    np.savez(output_dir / "split_indices.npz", **split_indices)

    metadata = dict(info)
    metadata.update(
        n_context=len(context_indices),
        n_test=len(test_indices),
        context_label_distribution=label_distribution(
            labels[context_indices], metadata["class_names"]
        ),
        test_label_distribution=label_distribution(
            labels[test_indices], metadata["class_names"]
        ),
        modalities=["image", "text"],
        modality_dims={"image": image_dim, "text": text_dim},
        structured_modalities=[],
        encoder=f"{image_description} ({image_dim}d) + {text_description} ({text_dim}d)",
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        normalization="none; raw encoder outputs retained without dataset-level normalization",
        normalization_fit_scope="none",
    )
    with (output_dir / "info.pkl").open("wb") as file:
        pickle.dump(metadata, file)
    print(f"Saved Raw features: {output_dir}", flush=True)
    return output_dir

