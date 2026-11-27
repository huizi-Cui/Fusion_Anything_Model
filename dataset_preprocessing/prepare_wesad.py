"""WESAD converts precomputed per-subject MOMENT NPZ files into 15 leave-one-subject-out folds."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from .dataset_paths import dataset_path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from .migration_common import save_dataset

CACHE_DIR = dataset_path('datasets/20_WESAD/data/wesad_moment60_noov')
OUTPUT_ROOT = ROOT / "data"
SUBJECTS = [f"S{x}" for x in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17)]
CLASSES = ["baseline", "stress", "amusement"]


def load_cache(cache_dir: Path):
    chest, wrist, labels, subjects, starts = [], [], [], [], []
    for subject in SUBJECTS:
        path = cache_dir / f"{subject}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing MOMENT cache: {path}")
        with np.load(path) as values:
            n = len(values["labels"])
            chest.append(values["chest"].astype(np.float32))
            wrist.append(values["wrist"].astype(np.float32))
            labels.append(values["labels"].astype(np.int64))
            starts.append(values["starts"].astype(np.int64))
            subjects.extend([subject] * n)
    return (np.concatenate(chest), np.concatenate(wrist), np.concatenate(labels),
            np.asarray(subjects), np.concatenate(starts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--pca48", action="store_true")
    args = parser.parse_args()
    chest, wrist, labels, subjects, starts = load_cache(args.cache_dir)
    counts = Counter(map(int, labels))
    for held_out in SUBJECTS:
        context = np.flatnonzero(subjects != held_out)
        test = np.flatnonzero(subjects == held_out)
        output = args.output_root / f"wesad_moment60_noov_loso_{held_out.lower()}"
        save_dataset(output, labels, context, test, {"chest": chest, "wrist": wrist}, {
            "display_name": f"WESAD LOSO {held_out}", "n_classes": 3,
            "class_names": CLASSES, "label_map": {1: 0, 2: 1, 3: 2},
            "modalities": ["chest", "wrist"], "structured_modalities": [],
            "encoder": "AutonLab/MOMENT-1-small; frozen; embedding task",
            "encoder_output_dim": 512,
            "encoder_input": "native-rate consecutive 512-sample blocks; mask partial block; mean-pool blocks",
            "split": f"LOSO; held-out subject {held_out}",
            "sample_unit": "60-second non-overlapping window within one subject/state run",
            "window_seconds": 60, "window_stride_seconds": 60,
            "window_counts_all": {CLASSES[k]: int(v) for k, v in counts.items()},
            "context_subjects": sorted(set(subjects[context])), "test_subjects": [held_out],
            "sample_ids": [f"{subjects[i]}:{int(starts[i])}" for i in np.r_[context, test]],
            "known_deviations_from_benchmark": [
                "user-approved non-overlap instead of official 0.25-second shift",
                "project-selected frozen MOMENT encoder",
            ],
            "source": str(dataset_path("datasets/20_WESAD/data/wesad/wesad.zip")),
        }, {"context": context, "test": test}, args.pca48)


if __name__ == "__main__":
    main()

