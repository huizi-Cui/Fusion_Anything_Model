"""Pinned source-tree and LoRA-checkpoint validation primitives."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal


PYTHON_HASH_PROBE = b"tabpfn-reproducibility-probe-v1"


def sha256_tree(root: Path) -> dict[str, Any]:
    """Hash a source tree by relative path and file bytes, ignoring caches."""

    root = root.resolve()
    digest = hashlib.sha256()
    file_count = 0
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or "__pycache__" in path.parts
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        file_count += 1
    return {
        "root": str(root),
        "file_count": file_count,
        "sha256": digest.hexdigest(),
    }


def sha256_files(root: Path, relative_paths: tuple[str, ...]) -> dict[str, Any]:
    """Hash an explicit, ordered source-file set below ``root``."""

    root = root.resolve()
    normalized = tuple(sorted(set(relative_paths)))
    if len(normalized) != len(relative_paths):
        raise ValueError("relative_paths must not contain duplicates")
    digest = hashlib.sha256()
    for relative_text in normalized:
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Path must stay below root: {relative_text}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        relative_bytes = relative.as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return {
        "root": str(root),
        "file_count": len(normalized),
        "files": list(normalized),
        "sha256": digest.hexdigest(),
    }


def verify_python_hash_seed(expected_seed: str = "0") -> dict[str, Any]:
    """Verify the interpreter hash secret, not only the environment variable."""

    configured_seed = os.environ.get("PYTHONHASHSEED")
    probe_literal = repr(PYTHON_HASH_PROBE)
    child_env = dict(os.environ)
    child_env["PYTHONHASHSEED"] = str(expected_seed)
    completed = subprocess.run(
        [sys.executable, "-c", f"print(hash({probe_literal}))"],
        env=child_env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    expected_probe = int(completed.stdout.strip())
    actual_probe = hash(PYTHON_HASH_PROBE)
    report = {
        "expected_seed": str(expected_seed),
        "configured_seed": configured_seed,
        "probe_input_hex": PYTHON_HASH_PROBE.hex(),
        "expected_probe": expected_probe,
        "actual_probe": actual_probe,
        "python_executable": sys.executable,
        "verified": configured_seed == str(expected_seed) and actual_probe == expected_probe,
    }
    if not report["verified"]:
        raise RuntimeError(
            "The active Python hash secret does not match PYTHONHASHSEED="
            f"{expected_seed}. Start a new process with PYTHONHASHSEED={expected_seed}; "
            "setting os.environ after interpreter startup is insufficient. "
            f"Probe report: {report}"
        )
    return report


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return the SHA256 digest of a file using bounded-memory reads."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def verify_sources(
    *,
    tabpfn_root: Path,
    local_wrapper_root: Path,
    wrapper_files: tuple[str, ...],
    expected_tabpfn_sha256: str,
    expected_wrapper_sha256: str,
    external_wrapper_policy: Literal["record", "require_match_if_present"],
) -> dict[str, Any]:
    """Validate imported sources while preserving the selected drift policy."""

    tabpfn_report = sha256_tree(tabpfn_root.resolve() / "src/tabpfn")
    if tabpfn_report["sha256"] != expected_tabpfn_sha256:
        raise RuntimeError("TabPFN source tree does not match the configured source hash")
    wrapper_report = sha256_files(local_wrapper_root, wrapper_files)
    if wrapper_report["sha256"] != expected_wrapper_sha256:
        raise RuntimeError("Bundled Fusion Anything model runtime has changed")
    source_wrapper_root = tabpfn_root.resolve() / "model_runtime"
    source_wrapper_report = (
        sha256_files(source_wrapper_root, wrapper_files)
        if source_wrapper_root.is_dir()
        else {"root": str(source_wrapper_root), "available": False}
    )
    if (
        external_wrapper_policy == "require_match_if_present"
        and source_wrapper_report.get("sha256")
        not in (None, wrapper_report["sha256"])
    ):
        raise RuntimeError("Bundled model runtime differs from its source copy")
    wrapper_report["external_copy"] = source_wrapper_report
    wrapper_report["external_copy_matches"] = (
        source_wrapper_report.get("sha256") == wrapper_report["sha256"]
    )
    return {"tabpfn": tabpfn_report, "wrapper": wrapper_report}


def checkpoint_metadata(
    *,
    path: Path,
    expected_sha256: str,
    step_contract: Literal["fixed_12000", "fixed_13000", "filename"],
) -> dict[str, Any]:
    """Validate the configured checkpoint hash, step, and rank-32 adapter contract."""

    import torch

    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"Unexpected LoRA checkpoint hash: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict")
    global_step = int(checkpoint.get("global_step", -1))
    if step_contract in {"fixed_12000", "fixed_13000"}:
        expected_step = 12000 if step_contract == "fixed_12000" else 13000
        if not isinstance(state, dict) or global_step != expected_step:
            raise ValueError(
                f"Expected a step-{expected_step} model_state_dict checkpoint"
            )
    else:
        filename_match = re.search(r"step_(\d+)\.pt$", path.name)
        filename_step = int(filename_match.group(1)) if filename_match else -1
        if (
            not isinstance(state, dict)
            or global_step < 0
            or filename_step != global_step
        ):
            raise ValueError(
                "Expected a model_state_dict checkpoint whose global_step "
                "matches its filename"
            )
    lora_keys = [key for key in state if key.endswith(".lora_A")]
    if (
        len(lora_keys) != 120
        or {int(state[key].shape[0]) for key in lora_keys} != {32}
    ):
        raise ValueError("Unexpected LoRA tensor count or rank")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": expected_sha256,
        "global_step": global_step,
        "state_dict_keys": len(state),
        "lora_rank": 32,
        "lora_alpha": 64,
    }
