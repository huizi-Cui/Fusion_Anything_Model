"""RAVDESS uses WavLM Base+ and DINOv3 ViT-B/16. Cache regeneration requires the external media extractor."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from .dataset_paths import dataset_path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from .migration_common import save_dataset

# Parent directory of the external inference_multimodal_1024d package.
EXTRACTOR_ROOT = Path("/path/to/feature_extractor")
RAW_ROOT = dataset_path('datasets/11_RAVDESS/data/ravdess')
CACHE_ROOT = dataset_path('datasets/11_RAVDESS/data/ravdess_cache')
OUTPUT_DIR = ROOT / "data/ravdess_wavlm_base_plus_dinov3_b"
CLASSES = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
TEST_ACTORS = {5, 6, 11, 12, 17, 18, 23, 24}


def rows():
    videos = {p.stem: p for p in sorted(RAW_ROOT.rglob("*.mp4")) if p.name.startswith("02-01-")}
    result = []
    for audio in sorted(RAW_ROOT.rglob("*.wav")):
        parts = audio.stem.split("-")
        if len(parts) != 7:
            continue
        modality, vocal, emotion, intensity, statement, repetition, actor = map(int, parts)
        if vocal != 1:
            continue
        video = videos.get(audio.stem.replace("03-", "02-", 1))
        if video is None:
            raise FileNotFoundError(f"Missing matched video-only clip for {audio}")
        result.append((audio, video, emotion - 1, actor))
    if not result:
        raise FileNotFoundError(f"No RAVDESS pairs under {RAW_ROOT}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_ROOT)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reextract", action="store_true")
    parser.add_argument("--pca48", action="store_true")
    args = parser.parse_args()
    data = rows()
    labels = np.asarray([x[2] for x in data], dtype=np.int64)
    actors = np.asarray([x[3] for x in data], dtype=np.int64)
    context = np.flatnonzero(~np.isin(actors, sorted(TEST_ACTORS)))
    test = np.flatnonzero(np.isin(actors, sorted(TEST_ACTORS)))
    audio_cache = args.cache_dir / "ravdess_wavlm_base_plus.npy"
    visual_cache = args.cache_dir / "ravdess_visual_video_only_dinov3_b_frames8.npy"
    if args.reextract or not (audio_cache.is_file() and visual_cache.is_file()):
        sys.path.insert(0, str(EXTRACTOR_ROOT))
        from inference_multimodal_1024d.prepare_extra_multimodal import extract_video_multiframe_features, extract_wavlm_features
        device = torch.device(args.device)
        audio = extract_wavlm_features([x[0] for x in data], device, "microsoft/wavlm-base-plus", 8, audio_cache)
        visual = extract_video_multiframe_features([x[1] for x in data], device, "dinov3_b", 32,
                                                    args.cache_dir / "ravdess_video_only_frames_8", 8, visual_cache)
    else:
        audio, visual = np.load(audio_cache), np.load(visual_cache)
    save_dataset(args.output_dir, labels, context, test, {"audio": audio, "visual": visual}, {
        "display_name": "RAVDESS", "n_classes": 8, "class_names": CLASSES,
        "modalities": ["audio", "visual"], "structured_modalities": [],
        "audio_encoder": "microsoft/wavlm-base-plus; temporal mean pooling", "audio_dim": 768,
        "visual_encoder": "DINOv3 ViT-B/16; mean of 8 uniformly sampled video-only frames", "visual_dim": 768,
        "split": f"fixed actor-disjoint; test actors={sorted(TEST_ACTORS)}; remaining actors Context",
        "sample_unit": "official audio-only WAV paired with official video-only MP4 speech utterance",
        "source": "https://zenodo.org/records/1188976",
    }, {"context": context, "test": test}, args.pca48)


if __name__ == "__main__":
    main()

