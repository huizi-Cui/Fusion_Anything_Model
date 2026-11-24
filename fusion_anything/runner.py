"""Evaluate extract_features PCA48 datasets with a selected LoRA checkpoint.

The default invocation performs preflight only.  GPU inference starts only when
``--run`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion_anything import assets as asset_core
from fusion_anything import inference as inference_core
from fusion_anything.data import GroupData, load_pca_dataset, load_registry, sha256_indices

DEFAULT_TABPFN_ROOT = PACKAGE_ROOT / "vendor/tabpfn_runtime"
DEFAULT_LORA_CHECKPOINT = PROJECT_ROOT / "checkpoints/FAM.pt"
CHECKPOINT_ENV_VAR = "FAM_CHECKPOINT"
HUGGINGFACE_REPO_URL = "https://huggingface.co/FusionAnything/FAMv1"
TREATMENT = "TL12_pca48_full"
EXPERIMENT_SEED = 42
LORA_SHA256 = "c479255355e66144a7b70195490d243a66a92fcfa85640187d5bb5a02e9d6e33"
TABPFN_SOURCE_SHA256 = "38394dac86a4c9a5764b8b88832f9e27a9a062762452f5cc8bc18b9fb848ed14"
CURRENT_WRAPPER_SHA256 = "2bb564093190d5e006aac8e5ac1786b1a160e0a4226eb0aa9c290d82d1fa02f9"
WRAPPER_FILES = ("__init__.py", "lora.py", "projector.py", "wrapper.py")
TABPFN_CONFIG = {
    "emsize": 192,
    "nlayers": 12,
    "nhead": 6,
    "features_per_group": 2,
}
MAX_NUM_CLASSES = 10
LORA_RANK = 32
LORA_ALPHA = 64


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a file without loading it all into memory."""
    return asset_core.sha256_file(path)


def verify_sources(tabpfn_root: Path) -> dict[str, Any]:
    """Check the pinned model and bundled TabPFN source hashes."""
    return asset_core.verify_sources(
        tabpfn_root=tabpfn_root,
        local_wrapper_root=PACKAGE_ROOT / "model",
        wrapper_files=WRAPPER_FILES,
        expected_tabpfn_sha256=TABPFN_SOURCE_SHA256,
        expected_wrapper_sha256=CURRENT_WRAPPER_SHA256,
        external_wrapper_policy="record",
    )


def configure_current_imports(tabpfn_root: Path) -> None:
    """Validate and add the bundled TabPFN runtime to the import path."""
    verify_sources(tabpfn_root)
    for value in (str(tabpfn_root.resolve() / "src"),):
        if value not in sys.path:
            sys.path.insert(0, value)


def resolve_checkpoint_path(checkpoint: str | Path | None = None) -> Path:
    """Resolve the FAM checkpoint from argument, environment variable, or default path."""
    if checkpoint is not None:
        return Path(checkpoint).expanduser()
    env_path = os.environ.get(CHECKPOINT_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser()
    return DEFAULT_LORA_CHECKPOINT


def checkpoint_metadata(
    path: Path, expected_sha256: str = LORA_SHA256,
) -> dict[str, Any]:
    """Validate the fixed 12k checkpoint and return its metadata."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"FAM checkpoint not found: {path}\n"
            f"Download 'FAM.pt' from {HUGGINGFACE_REPO_URL} "
            f"and place it at {DEFAULT_LORA_CHECKPOINT}, pass checkpoint='/path/to/checkpoint.pt', "
            f"or set {CHECKPOINT_ENV_VAR}."
        )
    return asset_core.checkpoint_metadata(
        path=path,
        expected_sha256=expected_sha256,
        step_contract="fixed_12000",
    )


def build_model(args: argparse.Namespace, device: Any) -> tuple[Any, dict[str, Any]]:
    """Construct the fixed model and load its validated full state.
    
    Args:
        args: Internal runtime configuration including source and checkpoint paths.
        device: Torch device on which to allocate the model.
    
    Returns:
        Evaluation-mode wrapper and checkpoint/runtime metadata.
    
    Notes:
        This is an internal constructor, not part of the public classifier API.
        Validate checkpoint integrity before constructing or loading the model."""
    import torch

    metadata = checkpoint_metadata(args.lora_checkpoint, args.lora_sha256)
    configure_current_imports(args.tabpfn_root)
    from fusion_anything.model.lora import apply_lora_to_tabpfn
    from fusion_anything.model.projector import UnstructuredModalityProcessor
    from fusion_anything.model.wrapper import TabPFNFineTuneWrapper
    from tabpfn.architectures.tabpfn_v2 import TabPFNV2, TabPFNV2Config

    config = TabPFNV2Config(**TABPFN_CONFIG)
    tabpfn = TabPFNV2(config=config, n_out=MAX_NUM_CLASSES)
    apply_lora_to_tabpfn(tabpfn, rank=LORA_RANK, alpha=LORA_ALPHA)
    wrapper = TabPFNFineTuneWrapper(
        tabpfn,
        UnstructuredModalityProcessor(embed_dim=tabpfn.embedding_dim),
        max_n_classes=MAX_NUM_CLASSES,
    ).to(device)
    wrapper._ensure_cls_head(device)
    checkpoint = torch.load(args.lora_checkpoint, map_location=device, weights_only=False)
    wrapper.load_state_dict(checkpoint["model_state_dict"], strict=True)
    wrapper.eval()
    return wrapper, {
        **metadata,
        "standalone_full_state": True,
        "tabpfn_config": {**TABPFN_CONFIG, "max_num_classes": MAX_NUM_CLASSES},
        "lora_config": {"rank": LORA_RANK, "alpha": LORA_ALPHA},
        "tabpfn_source_root": str(args.tabpfn_root.resolve()),
        "model_runtime": str((PACKAGE_ROOT / "model").resolve()),
    }


def preflight(
    data_root: Path, entries: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[dict[str, GroupData], dict[str, Any]]:
    """Validate prepared inputs and model assets without running inference."""
    datasets = {
        str(entry["name"]): load_pca_dataset(data_root, entry) for entry in entries
    }
    report = {}
    for entry in entries:
        name = str(entry["name"])
        data = datasets[name]
        report[name] = {
            "display_name": entry["display_name"],
            "data_dir": str(data.data_dir),
            "n_context": data.n_context,
            "n_test": data.n_test,
            "n_classes": data.n_classes,
            "n_features_concatenated": data.n_features,
            "feature_blocks": data.feature_blocks,
            "structured_modalities": entry.get("structured_modalities", []),
            "context_labels_sha256": sha256_indices(data.context_y),
            "test_labels_sha256": sha256_indices(data.test_y),
            "feature_variant": args.feature_variant,
        }
    sources = verify_sources(args.tabpfn_root)
    lora = checkpoint_metadata(args.lora_checkpoint, args.lora_sha256)
    payload = {
        "status": "preflight_complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "treatment": args.treatment,
        "models": [f"TL_lora_step_{lora['global_step']}"],
        "finetuned": {
            "test_query_batch_size": 128,
            "experiment_seed": EXPERIMENT_SEED,
            "inference_feature_shuffle": False,
        },
        "pythonhashseed": 42,
        "variant": f"{args.feature_variant}_only",
        "full_context_no_sampling": True,
        "data_root": str(data_root),
        "dataset_order": [str(entry["name"]) for entry in entries],
        "datasets": report,
        "model_architecture": {
            "tabpfn_config": {**TABPFN_CONFIG, "max_num_classes": MAX_NUM_CLASSES},
            "lora_config": {"rank": LORA_RANK, "alpha": LORA_ALPHA},
            "standalone_full_state": True,
        },
        "lora_checkpoint": lora,
        "sources": sources,
    }
    return datasets, payload


def main() -> int:
    """Run the internal evaluator CLI; infer only when --run is supplied."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--tabpfn-root", type=Path, default=DEFAULT_TABPFN_ROOT)
    parser.add_argument("--lora-checkpoint", type=Path, default=DEFAULT_LORA_CHECKPOINT)
    parser.add_argument("--lora-sha256", default=LORA_SHA256)
    parser.add_argument("--treatment", default=TREATMENT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="float32")
    parser.add_argument("--finetuned-chunk-size", type=int, default=128)
    parser.add_argument("--invariance-atol", type=float, default=5e-4)
    parser.add_argument("--run", action="store_true", help="start GPU inference after preflight")
    args = parser.parse_args()
    args.feature_variant = "pca48"
    args.output_root = args.output_root.resolve()
    args.enforce_query_chunk_invariance = False

    data_root, entries = load_registry(args.registry)
    args.display_names = {str(entry["name"]): str(entry["display_name"]) for entry in entries}
    datasets, preflight_payload = preflight(data_root, entries, args)
    inference_core.write_json(args.output_root / "preflight.json", preflight_payload)
    print(json.dumps(preflight_payload, indent=2, ensure_ascii=False))
    if not args.run:
        print("PREFLIGHT ONLY: add --run after review to start GPU inference")
        return 0

    import torch
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA-visible execution is required")
    torch.cuda.set_device(device)
    torch.cuda.init()
    if args.dtype == "float32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    args.python_hash_report = asset_core.verify_python_hash_seed("42")
    np.random.seed(EXPERIMENT_SEED)
    random.seed(EXPERIMENT_SEED)
    torch.manual_seed(EXPERIMENT_SEED)
    torch.cuda.manual_seed_all(EXPERIMENT_SEED)
    records = []
    wrapper, manifest = build_model(args, device)
    if args.dtype == "float32":
        parameter_dtypes = sorted({
            str(parameter.dtype)
            for parameter in wrapper.parameters()
            if parameter.is_floating_point()
        })
        if parameter_dtypes != ["torch.float32"]:
            raise RuntimeError(f"Fine wrapper parameter dtype drift: {parameter_dtypes}")
        manifest["precision_scope"] = "complete_multimodal_wrapper"
    manifest["pythonhashseed"] = os.environ.get("PYTHONHASHSEED")
    manifest["python_hash_runtime_probe"] = args.python_hash_report
    manifest["random_seed"] = EXPERIMENT_SEED
    manifest["inference_feature_shuffle"] = False
    for entry in entries:
        name = str(entry["name"])
        types = [
            "structured" if modality in set(entry.get("structured_modalities") or []) else "unstructured"
            for modality in entry["modalities"]
        ]
        records.append(inference_core.run_finetuned(
            wrapper=wrapper, checkpoint_manifest=manifest, dataset=name,
            variant=args.feature_variant, data=datasets[name], types=types, args=args,
        ))
    del wrapper
    gc.collect()
    torch.cuda.empty_cache()
    summary = inference_core.summarize(records, args.output_root, args.treatment)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
