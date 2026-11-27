"""UR-FUNNY GloVe, COVAREP and OpenFace pooling. Legacy LSTM mode also needs BERT and the external encoders package."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from .dataset_paths import dataset_path
import numpy as np
import pickle
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from .encoders_zero import split_without_normalization

DEFAULT_DATA_DIR = dataset_path('datasets/07_UR-FUNNY/data/UR-FUNNY/urfunny2_features')
OUTPUT_DIR = ROOT / "data/urfunny_zero"
# For legacy_lstm, add the parent of the external encoders package to PYTHONPATH.
BATCH_SIZE = 64

# 仅 legacy_lstm 模式使用。mean_std 模式直接对完整变长序列聚合。
MAX_SEQ_LEN = 240


def pool_sequence(seq: np.ndarray, mode: str) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32)
    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
    if seq.ndim != 2:
        raise ValueError(f"Expected [T,D] sequence, got shape={seq.shape}")
    if seq.shape[0] == 0:
        mean = np.zeros(seq.shape[1], dtype=np.float32)
        std = np.zeros(seq.shape[1], dtype=np.float32)
    else:
        mean = seq.mean(axis=0)
        std = seq.std(axis=0)
    if mode == "mean":
        return mean.astype(np.float32)
    return np.concatenate([mean, std], axis=0).astype(np.float32)


def lookup_text_sequence(lang_item: dict, embedding_table: np.ndarray) -> np.ndarray:
    indexes = []
    for sent_indexes in lang_item["context_embedding_indexes"]:
        indexes.extend(sent_indexes)
    indexes.extend(lang_item["punchline_embedding_indexes"])
    if not indexes:
        return np.zeros((1, embedding_table.shape[1]), dtype=np.float32)
    idx = np.asarray(indexes, dtype=np.int64)
    idx = np.clip(idx, 0, len(embedding_table) - 1)
    return embedding_table[idx]


def concat_feature_sequence(feat_item: dict) -> np.ndarray:
    frames = [arr.astype(np.float32) for arr in feat_item["context_features"]]
    frames.append(feat_item["punchline_features"].astype(np.float32))
    return np.concatenate(frames, axis=0)


def load_urfunny_data(data_dir: Path, pooling: str, max_seq_len: int = MAX_SEQ_LEN):
    '加载 UR-FUNNY 数据，返回合并后的特征和标签。.'
    print("  加载数据文件...")
    with open(data_dir / "data_folds.pkl", "rb") as f:
        folds = pickle.load(f)
    with open(data_dir / "humor_label_sdk.pkl", "rb") as f:
        labels_dict = pickle.load(f)
    with open(data_dir / "language_sdk.pkl", "rb") as f:
        language = pickle.load(f)
    with open(data_dir / "covarep_features_sdk.pkl", "rb") as f:
        covarep = pickle.load(f)
    with open(data_dir / "openface_features_sdk.pkl", "rb") as f:
        openface = pickle.load(f)
    with open(data_dir / "word_embedding_list.pkl", "rb") as f:
        embedding_table = np.asarray(pickle.load(f), dtype=np.float32)

    # 官方划分: train + dev → context, test → test
    ctx_ids = folds["train"] + folds["dev"]
    test_ids = folds["test"]
    all_ids = ctx_ids + test_ids
    n_ctx = len(ctx_ids)
    n_test = len(test_ids)
    n_total = len(all_ids)

    print(f"  Train IDs: {len(folds['train'])}, Dev IDs: {len(folds['dev'])}, Test IDs: {len(folds['test'])}")
    print(f"  Context (train+dev): {n_ctx}, Test: {n_test}, Total: {n_total}")

    # 检查缺失
    for name, feat_dict in [("covarep", covarep), ("openface", openface),
                             ("language", language), ("labels", labels_dict)]:
        missing = [i for i in all_ids if i not in feat_dict]
        if missing:
            print(f"  ⚠  {name}: {len(missing)} IDs 缺失")
    assert len(missing) == 0, f"存在缺失ID: {missing[:5]}..."


    texts = []
    text_pooled = []
    audio_pooled = []
    visual_pooled = []
    audio_list = []   # legacy_lstm: list of np.ndarray (T_i, 81)
    visual_list = []  # legacy_lstm: list of np.ndarray (T_i, 371)
    labels = np.array([labels_dict[i] for i in all_ids], dtype=np.int64)

    max_text_t = 0
    max_audio_t = 0
    max_visual_t = 0
    for vid in all_ids:
        # Text: 拼接上下文句子 + 笑点句子
        lang = language[vid]
        context_text = " ".join(lang["context_sentences"])
        punchline_text = lang["punchline_sentence"]
        full_text = context_text + " [SEP] " + punchline_text
        texts.append(full_text)

        # Audio: 拼接 context + punchline COVAREP 帧
        text_seq = lookup_text_sequence(lang, embedding_table)
        a_seq = concat_feature_sequence(covarep[vid])  # (T_a, 81)
        v_seq = concat_feature_sequence(openface[vid])  # (T_v, 371)

        max_text_t = max(max_text_t, text_seq.shape[0])
        max_audio_t = max(max_audio_t, a_seq.shape[0])
        max_visual_t = max(max_visual_t, v_seq.shape[0])

        if pooling in ("mean_std", "mean"):
            text_pooled.append(pool_sequence(text_seq, pooling))
            audio_pooled.append(pool_sequence(a_seq, pooling))
            visual_pooled.append(pool_sequence(v_seq, pooling))
        else:
            audio_list.append(a_seq)
            visual_list.append(v_seq)

    print(f"  Text 样本数: {len(texts)}")
    print(f"  原始最大长度: Text={max_text_t}, Audio={max_audio_t}, Vision={max_visual_t}")

    if pooling in ("mean_std", "mean"):
        text_feats = np.stack(text_pooled).astype(np.float32)
        audio_feats = np.stack(audio_pooled).astype(np.float32)
        visual_feats = np.stack(visual_pooled).astype(np.float32)
        print(f"  Pooling={pooling}: Text={text_feats.shape}, Audio={audio_feats.shape}, Vision={visual_feats.shape}")
    else:
        text_feats = None
        print(f"  legacy_lstm 截断/补齐长度: {max_seq_len}")
        audio_feats = np.zeros((n_total, max_seq_len, audio_list[0].shape[1]), dtype=np.float32)
        visual_feats = np.zeros((n_total, max_seq_len, visual_list[0].shape[1]), dtype=np.float32)
        for i, (a_seq, v_seq) in enumerate(zip(audio_list, visual_list)):
            a_len = min(a_seq.shape[0], max_seq_len)
            v_len = min(v_seq.shape[0], max_seq_len)
            audio_feats[i, :a_len] = a_seq[:a_len]
            visual_feats[i, :v_len] = v_seq[:v_len]

        n_truncated_audio = sum(1 for a in audio_list if a.shape[0] > max_seq_len)
        n_truncated_visual = sum(1 for v in visual_list if v.shape[0] > max_seq_len)
        print(f"  Audio 截断样本: {n_truncated_audio}/{n_total}, Visual 截断样本: {n_truncated_visual}/{n_total}")

    ctx_idx = np.arange(n_ctx)
    test_idx = np.arange(n_ctx, n_total)

    # 标签分布
    for name, idx in [("Context", ctx_idx), ("Test", test_idx)]:
        bc = np.bincount(labels[idx])
        print(f"  {name} 标签: 0={bc[0]}, 1={bc[1]} (balance={bc[0]/bc[1]:.2f})")

    return texts, text_feats, audio_feats, visual_feats, labels, ctx_idx, test_idx


def extract_legacy_lstm_features(texts, audio_feats, visual_feats, device, batch_size):
    from .encoders_zero import (
        build_sequence_lstm_encoder, extract_sequence_lstm_features, extract_bert_features,
    )

    print("\n=== 构建 legacy 编码器 ===")
    lstm_aud = build_sequence_lstm_encoder(device, input_dim=81, hidden_dim=256, num_layers=2)
    lstm_vis = build_sequence_lstm_encoder(device, input_dim=371, hidden_dim=512, num_layers=2)

    print("\n--- Text (BERT → 768d) ---")
    txt_feats = extract_bert_features(texts, device, batch_size=batch_size)
    print(f"  {txt_feats.shape}")

    print("\n--- Audio (SeqLSTM 81→256) ---")
    aud_feats = extract_sequence_lstm_features(lstm_aud, audio_feats, device, batch_size=batch_size)
    print(f"  {aud_feats.shape}")

    print("\n--- Vision (SeqLSTM 371→512) ---")
    vis_feats = extract_sequence_lstm_features(lstm_vis, visual_feats, device, batch_size=batch_size)
    print(f"  {vis_feats.shape}")
    return txt_feats, aud_feats, vis_feats


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare UR-FUNNY for LimiX multimodal inference.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Path to UR-FUNNY urfunny2_features directory.")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR,
                        help="Output directory for context/test npy files.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Feature extraction batch size.")
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN,
                        help="Max sequence length for legacy_lstm audio/visual features.")
    parser.add_argument("--pooling", choices=["mean_std", "mean", "legacy_lstm"], default="mean_std",
                        help="How to convert official UR-FUNNY sequences to fixed vectors.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n=== 加载 UR-FUNNY ===")
    print(f"Source: {args.data_dir}")
    texts, text_feats, audio_feats, visual_feats, labels, ctx_idx, test_idx = \
        load_urfunny_data(args.data_dir, pooling=args.pooling, max_seq_len=args.max_seq_len)
    n_total = len(labels)
    print(f"Total: {n_total}, Context: {len(ctx_idx)}, Test: {len(test_idx)}")
    print(f"Labels: 0={np.sum(labels==0)}, 1={np.sum(labels==1)}")


    if args.pooling == "legacy_lstm":
        txt_feats, aud_feats, vis_feats = extract_legacy_lstm_features(
            texts, audio_feats, visual_feats, device, args.batch_size
        )
        encoder_desc = "BERT(768d)+SeqLSTM(81→256)+SeqLSTM(371→512)"
    else:
        txt_feats, aud_feats, vis_feats = text_feats, audio_feats, visual_feats
        encoder_desc = f"official GloVe/COVAREP/OpenFace + {args.pooling} temporal pooling"


    print("\n=== Raw Context/Test Split ===")
    txt_ctx, txt_test = split_without_normalization(txt_feats, ctx_idx, test_idx)
    aud_ctx, aud_test = split_without_normalization(aud_feats, ctx_idx, test_idx)
    vis_ctx, vis_test = split_without_normalization(vis_feats, ctx_idx, test_idx)


    print("\n=== 保存 ===")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "context_text.npy", txt_ctx)
    np.save(output_dir / "test_text.npy", txt_test)
    np.save(output_dir / "context_audio.npy", aud_ctx)
    np.save(output_dir / "test_audio.npy", aud_test)
    np.save(output_dir / "context_vision.npy", vis_ctx)
    np.save(output_dir / "test_vision.npy", vis_test)
    np.save(output_dir / "context_labels.npy", labels[ctx_idx])
    np.save(output_dir / "test_labels.npy", labels[test_idx])
    np.save(output_dir / "context_indices.npy", np.arange(len(ctx_idx), dtype=np.int64))
    np.savez(output_dir / "split_indices.npz", context=ctx_idx, test=test_idx)

    import pickle as pkl
    with open(output_dir / "info.pkl", "wb") as f:
        pkl.dump({
            "dataset": "urfunny_zero",
            "n_classes": 2,
            "n_context": len(ctx_idx), "n_test": len(test_idx),
            "text_dim": txt_ctx.shape[1],
            "audio_dim": aud_ctx.shape[1],
            "vision_dim": vis_ctx.shape[1],
            "encoder": encoder_desc,
            "pooling": args.pooling,
            "class_names": ["not_funny", "funny"],
            "max_seq_len": args.max_seq_len,
            "normalization": "none; pooled features retained without dataset-level normalization",
            "normalization_fit_scope": "none",
        }, f)

    print(f"\nDone: {output_dir}")
    print(f"  Text: {txt_ctx.shape[1]}d | Audio: {aud_ctx.shape[1]}d | Vision: {vis_ctx.shape[1]}d")
    print(f"  Context: {len(ctx_idx)}, Test: {len(test_idx)}, Classes: 2")


if __name__ == "__main__":
    main()

