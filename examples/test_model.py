"""Evaluate a prepared Derm7pt or MultiOFF example with the fixed 12k model."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LORA = PROJECT_ROOT / "checkpoints/FAM.safetensors"
LORA_SHA256 = "bb17b1629e36a19c0bded3fd9ecc508c7440f1e127df2af823f99d3b73159c5b"


EXAMPLES = {
    "derm7pt": {"directory": "Derm7pt", "task": "derm7pt_zero_pca",
                "display": "Derm7pt", "modalities": ["image", "tabular"],
                "structured": ["tabular"], "output": "derm7pt_12k"},
    "multioff": {"directory": "MultiOFF", "task": "multioff_zero_pca",
                "display": "MultiOFF", "modalities": ["image", "text"],
                "structured": [], "output": "multioff_12k"},
}


def build_parser() -> argparse.ArgumentParser:
    """Define one evaluator with explicit presets for the two examples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=tuple(EXAMPLES), default="derm7pt")
    parser.add_argument("--data-dir", type=Path,
                        default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--modalities", nargs="+", default=None)
    parser.add_argument("--structured-modalities", nargs="*", default=None)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(os.environ["FAM_CHECKPOINT"]).expanduser()
        if os.environ.get("FAM_CHECKPOINT") else LORA,
        help="Path to FAM.safetensors (defaults to FAM_CHECKPOINT or checkpoints/)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def resolve_configuration(args: argparse.Namespace) -> argparse.Namespace:
    """Fill only unspecified options from the selected example preset.

    Empty --structured-modalities explicitly overrides any preset declaration.
    The MultiOFF preset has no structured modality; text is not treated as table.
    """
    preset = EXAMPLES[args.example]
    defaults = {
        "data_dir": PROJECT_ROOT / "datasets" / preset["directory"] / "processed" / preset["task"],
        "output_dir": PROJECT_ROOT / "outputs" / preset["output"],
        "dataset_name": preset["task"], "display_name": preset["display"],
        "modalities": list(preset["modalities"]),
        "structured_modalities": list(preset["structured"]),
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def main(argv: list[str] | None = None) -> int:
    """Validate or evaluate the selected example using the shared classifier."""
    args = resolve_configuration(build_parser().parse_args(argv))

    data_dir = args.data_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry = output / "input_registry.json"
    registry.write_text(json.dumps({
        "data_root": str(data_dir.parent),
        "datasets": [{
            "name": args.dataset_name,
            "display_name": args.display_name,
            "data_dir": str(data_dir),
            "modalities": args.modalities,
            "structured_modalities": args.structured_modalities,
        }],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    command = [
        sys.executable, "-m", "fusion_anything.runner",
        "--registry", str(registry),
        "--output-root", str(output),
        "--device", args.device,
        "--dtype", "float32",
        "--finetuned-chunk-size", "128",
        "--lora-checkpoint", str(args.checkpoint.expanduser().resolve()),
        "--lora-sha256", LORA_SHA256,
        "--treatment", "TL12_pca48_full",
    ]
    if not args.preflight_only:
        command.append("--run")
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "42"
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
