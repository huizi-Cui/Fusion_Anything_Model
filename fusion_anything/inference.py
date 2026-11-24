"""Internal fixed-weight inference, evaluation metrics, and result persistence."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from fusion_anything.data import GroupData, label_distribution, sha256_indices


DEFAULT_TREATMENT = "tabpfn_internal_full_context_raw_pca48_v1"


def jsonable(value: Any) -> Any:
    """Convert evaluation values into JSON-compatible objects."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write an indented UTF-8 JSON artifact with a trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def compute_ece(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10,
) -> float:
    """Compute confidence-based expected calibration error over equal-width bins."""
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    correct = predictions == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            value += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidences[mask].mean())
            )
    return float(value)


def compute_metrics(
    y_true: np.ndarray, probabilities: np.ndarray,
) -> dict[str, Any]:
    """Compute classification metrics from Query labels and probabilities.
    
    Query labels are used only here, after predictions have been produced.
    Multiclass AUROC uses one-vs-rest with macro averaging.
    AUROC is None when the provided label distribution cannot define it."""
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    classes = np.arange(probabilities.shape[1], dtype=np.int64)
    precision, recall, per_f1, support = precision_recall_fscore_support(
        y_true, predictions, labels=classes, zero_division=0,
    )
    try:
        if len(classes) == 2:
            auc = float(roc_auc_score(y_true, probabilities[:, 1]))
        else:
            auc = float(roc_auc_score(
                y_true, probabilities, multi_class="ovr", average="macro",
            ))
    except ValueError:
        auc = None
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted")),
        "auroc": auc,
        "auc": auc,
        "log_loss": float(log_loss(y_true, probabilities, labels=classes)),
        "ece": compute_ece(y_true, probabilities),
        "prediction_distribution": label_distribution(predictions),
        "per_class_metrics": {
            str(int(cls)): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(per_f1[index]),
                "support": int(support[index]),
            }
            for index, cls in enumerate(classes)
        },
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=classes,
        ).tolist(),
    }


def split_blocks(data: GroupData, values: np.ndarray) -> dict[str, np.ndarray]:
    """Split concatenated features into the stored modality column ranges."""
    output: dict[str, np.ndarray] = {}
    for block in data.feature_blocks:
        start = int(block["column_start"])
        end = int(block["column_end_exclusive"])
        output[str(block["name"])] = np.asarray(
            values[:, start:end], dtype=np.float32
        )
    return output


def _save_result(
    *,
    output_dir: Path,
    payload: dict[str, Any],
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    np.save(
        output_dir / "context_indices.npy",
        np.arange(payload["data"]["n_context"], dtype=np.int64),
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        probabilities=probabilities,
        predictions=predictions,
        labels=labels,
    )
    path = output_dir / "result.json"
    write_json(path, payload)
    return path


def _fine_tuned_probabilities(
    *,
    wrapper: Any,
    data: GroupData,
    types: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run fixed-weight inference using all Context rows in each Query chunk.
    
    Real Query labels are never read by this engine. Zero labels satisfy the
    wrapper loss signature; the loss is discarded. Reduce chunk size on CUDA
    memory exhaustion without subsampling Context."""
    import torch

    device = torch.device(args.device)
    context_blocks = split_blocks(data, data.context_x)
    test_blocks = split_blocks(data, data.test_x)
    names = list(context_blocks)
    if len(types) != len(names):
        raise RuntimeError("Modality type count does not match modality blocks")
    tabular_dim = sum(
        int(context_blocks[name].shape[1])
        for name, modality_type in zip(names, types, strict=True)
        if modality_type == "structured"
    )
    context_tensors = {
        name: torch.from_numpy(values).to(device)
        for name, values in context_blocks.items()
    }
    test_tensors = {
        name: torch.from_numpy(values).to(device)
        for name, values in test_blocks.items()
    }
    context_y = torch.from_numpy(
        np.array(data.context_y, dtype=np.int64, copy=True)
    ).to(device)
    # Query labels are placeholders used only by the wrapper's unused loss.
    test_y = torch.zeros(data.n_test, dtype=torch.int64, device=device)
    if args.dtype == "bfloat16":
        amp_dtype = torch.bfloat16
    elif args.dtype == "float16":
        amp_dtype = torch.float16
    elif args.dtype == "float32":
        amp_dtype = None
    else:
        raise ValueError(f"Unsupported inference dtype: {args.dtype}")

    def forward(start: int, end: int) -> np.ndarray:
        modality_data = {
            name: torch.cat(
                [context_tensors[name], test_tensors[name][start:end]], dim=0
            ).unsqueeze(0)
            for name in names
        }
        labels = torch.cat([context_y, test_y[start:end]], dim=0).unsqueeze(0)
        expected_rows = data.n_context + (end - start)
        if any(
            int(values.shape[1]) != expected_rows
            for values in modality_data.values()
        ):
            raise RuntimeError("Fine-tuned forward did not receive full Context")
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None
        ):
            output = wrapper(
                modality_data=modality_data,
                y=labels,
                train_pos=data.n_context,
                modality_names=names,
                modality_types=types,
                modality_dims=[
                    int(context_blocks[name].shape[1]) for name in names
                ],
                tabular_dim=tabular_dim,
                dataset_K=24,
                n_classes=data.n_classes,
            )
        logits = output["cls_output_current"].squeeze(0)
        if args.dtype == "float32" and logits.dtype != torch.float32:
            raise RuntimeError(f"Fine FP32 logits unexpectedly use {logits.dtype}")
        logits = logits.float()
        values = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
        del modality_data, labels, output, logits
        return values

    chunk_size = min(args.finetuned_chunk_size, data.n_test)
    invariance_attempts: list[dict[str, Any]] = []
    enforce_invariance = bool(
        getattr(args, "enforce_query_chunk_invariance", True)
    )
    if enforce_invariance:
        check_n = min(4, data.n_test)
        single_check = np.concatenate(
            [forward(i, i + 1) for i in range(check_n)], axis=0
        )
        while chunk_size > 1:
            compare_n = min(check_n, chunk_size)
            batch_check = forward(0, chunk_size)[:compare_n]
            reference = single_check[:compare_n]
            max_abs = float(np.max(np.abs(batch_check - reference)))
            argmax_equal = bool(
                np.array_equal(
                    batch_check.argmax(axis=1), reference.argmax(axis=1)
                )
            )
            invariance_attempts.append(
                {
                    "query_chunk_size": chunk_size,
                    "max_abs": max_abs,
                    "argmax_equal": argmax_equal,
                }
            )
            del batch_check
            if max_abs <= args.invariance_atol and argmax_equal:
                break
            chunk_size = max(1, chunk_size // 2)
        if chunk_size == 1:
            invariance_max_abs = 0.0
            invariance_attempts.append(
                {"query_chunk_size": 1, "max_abs": 0.0, "argmax_equal": True}
            )
        else:
            invariance_max_abs = float(invariance_attempts[-1]["max_abs"])
        del single_check
    else:
        check_n = 0
        invariance_max_abs = None
    torch.cuda.synchronize(device)

    while True:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            parts = [
                forward(start, min(start + chunk_size, data.n_test))
                for start in range(0, data.n_test, chunk_size)
            ]
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            probabilities = np.concatenate(parts, axis=0)
            peak = int(torch.cuda.max_memory_allocated(device))
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if chunk_size == 1:
                raise
            chunk_size = max(1, chunk_size // 2)
            print(
                f"OOM retry with fine-tuned query chunk={chunk_size}", flush=True
            )

    row_sums = probabilities.sum(axis=1)
    max_deviation = float(np.max(np.abs(row_sums - 1.0)))
    probabilities /= row_sums[:, None]
    del context_tensors, test_tensors, context_y, test_y
    torch.cuda.empty_cache()
    return probabilities, {
        "predict_seconds": elapsed,
        "peak_gpu_memory_bytes": peak,
        "query_chunk_size": chunk_size,
        "query_chunk_invariance_check_rows": check_n,
        "query_chunk_invariance_enforced": enforce_invariance,
        "query_chunk_invariance_max_abs": invariance_max_abs,
        "query_chunk_invariance_atol": args.invariance_atol,
        "query_chunk_invariance_attempts": invariance_attempts,
        "max_abs_row_sum_deviation_before_renormalization": max_deviation,
        "full_context_rows_per_forward": data.n_context,
        "modality_types": types,
        "tabular_dim": tabular_dim,
        "autocast_enabled": amp_dtype is not None,
        "tf32_enabled": bool(
            torch.backends.cuda.matmul.allow_tf32
            or torch.backends.cudnn.allow_tf32
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def run_finetuned(
    *,
    wrapper: Any,
    checkpoint_manifest: dict[str, Any],
    dataset: str,
    variant: str,
    data: GroupData,
    types: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Predict through the classifier, then compute metrics and persist results."""
    from fusion_anything.classifier import FusionAnythingClassifier

    classifier = FusionAnythingClassifier(device=args.device, chunk_size=args.finetuned_chunk_size)
    classifier.fit(
        split_blocks(data, data.context_x), data.context_y,
        structured_modalities=[
            str(block["name"]) for block, modality_type in zip(data.feature_blocks, types, strict=True)
            if modality_type == "structured"
        ],
    )
    classifier._wrapper = wrapper
    classifier._inference_args = args
    probabilities = classifier.predict_proba(split_blocks(data, data.test_x))
    runtime = classifier.runtime_
    if probabilities.shape != (data.n_test, data.n_classes):
        raise RuntimeError(
            f"{dataset}/{variant}: unexpected fine-tuned probability shape"
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            f"{dataset}/{variant}: non-finite fine-tuned probabilities"
        )
    metrics = compute_metrics(data.test_y, probabilities)
    step = int(checkpoint_manifest["global_step"])
    route = f"TL{step // 1000}_{variant}_full"
    payload = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "treatment": args.treatment,
        "dataset": dataset,
        "display_name": args.display_names[dataset],
        "route": route,
        "diagnostic_or_formal": "diagnostic_internal_tabpfn_comparison",
        "model": checkpoint_manifest,
        "data": {
            "variant": variant,
            "data_dir": str(data.data_dir),
            "n_context": data.n_context,
            "n_test": data.n_test,
            "n_classes": data.n_classes,
            "modality_blocks": data.feature_blocks,
            "modality_types": types,
            "full_context_no_sampling": True,
            "context_indices_sha256": sha256_indices(
                np.arange(data.n_context, dtype=np.int64)
            ),
            "all_modalities_unstructured": all(
                value == "unstructured" for value in types
            ),
            "cap_tokens_per_modality": 24,
            "all_context_rows_passed_to_every_forward": True,
            "all_input_features_used_by_structured_or_mgm_path": True,
        },
        "runtime": {"device": args.device, "dtype": args.dtype, **runtime},
        "probability_postprocessing": {
            "renormalized_to_unit_sum": True,
            "max_abs_row_sum_deviation_before_renormalization": runtime[
                "max_abs_row_sum_deviation_before_renormalization"
            ],
        },
        "metrics": metrics,
    }
    result_path = _save_result(
        output_dir=args.output_root / dataset / route,
        payload=payload,
        probabilities=probabilities,
        labels=data.test_y,
    )
    print(
        f"COMPLETE {dataset} {route} Acc={metrics['accuracy']:.4f} "
        f"AUROC={metrics['auroc']} MacroF1={metrics['macro_f1']:.4f}",
        flush=True,
    )
    return {"path": str(result_path), "payload": payload}


def summarize(
    records: list[dict[str, Any]],
    output_root: Path,
    treatment: str = DEFAULT_TREATMENT,
) -> dict[str, Any]:
    """Collect completed task records and write the evaluation summary."""
    preflight_path = output_root / "preflight.json"
    preflight_payload = (
        json.loads(preflight_path.read_text(encoding="utf-8"))
        if preflight_path.is_file()
        else {}
    )
    rows = []
    for record in records:
        payload = record["payload"]
        row = {
            "status": payload["status"],
            "dataset": payload["dataset"],
            "display_name": payload["display_name"],
            "route": payload["route"],
            "n_context": payload["data"]["n_context"],
            "n_test": payload["data"]["n_test"],
            "result_path": record["path"],
        }
        if payload["status"] == "complete":
            row["metrics"] = payload["metrics"]
        else:
            row["failure"] = payload["failure"]
        rows.append(row)
    n_complete = sum(row["status"] == "complete" for row in rows)
    n_skipped_resource = sum(
        row["status"] == "skipped_resource_exhausted" for row in rows
    )
    n_skipped_model_limit = sum(
        row["status"] == "skipped_model_input_limit" for row in rows
    )
    n_skipped = n_skipped_resource + n_skipped_model_limit
    summary = {
        "status": "complete" if n_skipped == 0 else "complete_with_skips",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "treatment": treatment,
        "comparison_scope": "standalone Fusion Anything v1; full Context",
        "n_complete": n_complete,
        "n_skipped_resource_exhausted": n_skipped_resource,
        "n_skipped_model_input_limit": n_skipped_model_limit,
        "n_registered_prepared_split_rows": len(preflight_payload.get("data", {})),
        "unavailable_or_requires_preparation": preflight_payload.get(
            "unavailable_or_requires_preparation", []
        ),
        "rows": rows,
    }
    write_json(output_root / "summary.json", summary)
    return summary
