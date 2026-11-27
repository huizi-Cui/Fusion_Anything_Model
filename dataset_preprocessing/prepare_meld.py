"""MELD uses precomputed MultiBench GloVe text and audio features."""

from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path
from .dataset_paths import dataset_path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from .encoders_zero import split_without_normalization
from .prepare_projection import project_dataset

FEATURE_DIR=dataset_path('datasets/19_MELD/data/MELD/features')
OUTPUT_DIR=ROOT/"data/meld_zero"
FILES={"metadata":"data_sentiment.p","text":"text_glove_average_sentiment.pkl","audio":"audio_embeddings_feature_selection_sentiment.pkl"}
SEED=42

def load_pickle(path):
    with Path(path).open("rb") as file:
        try:return pickle.load(file)
        except UnicodeDecodeError:file.seek(0);return pickle.load(file,encoding="latin1")

def normalize_split(value):
    value=str(value).strip().lower()
    if value in {"val","valid","validation","dev"}:return "val"
    if value in {"train","test"}:return value
    raise ValueError(f"Unknown MELD split: {value}")

def feature(splits,split,dialog,utterance):
    values=splits[{"train":0,"val":1,"test":2}[split]]; key=f"{dialog}_{utterance}"
    if key in values:return np.asarray(values[key],dtype=np.float32)
    dialog_values=values.get(dialog,values.get(int(dialog) if str(dialog).isdigit() else dialog))
    if dialog_values is None:raise KeyError(f"Missing MELD feature {split}/{key}")
    dialog_values=np.asarray(dialog_values); return np.asarray(dialog_values if dialog_values.ndim==1 else dialog_values[int(utterance)],dtype=np.float32)

def load_data():
    metadata=load_pickle(FEATURE_DIR/FILES["metadata"]); rows=metadata[0]; label_index={str(k):int(v) for k,v in metadata[5].items()}
    text_splits=load_pickle(FEATURE_DIR/FILES["text"]); audio_splits=load_pickle(FEATURE_DIR/FILES["audio"])
    texts=[]; audios=[]; labels=[]; context=[]; test=[]
    for row in rows:
        split=normalize_split(row["split"]); index=len(labels); dialog=str(row["dialog"]); utterance=int(row["utterance"])
        texts.append(feature(text_splits,split,dialog,utterance)); audios.append(feature(audio_splits,split,dialog,utterance))
        value=row.get("y",row.get("label")); labels.append(label_index[str(value)] if str(value) in label_index else int(value))
        (context if split in {"train","val"} else test).append(index)
    names=[name for name,_ in sorted(label_index.items(),key=lambda item:item[1])]
    return np.stack(texts),np.stack(audios),np.asarray(labels,dtype=np.int64),np.asarray(context),np.asarray(test),label_index,names

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir",type=Path,default=OUTPUT_DIR); parser.add_argument("--pca48",action="store_true"); args=parser.parse_args()
    text,audio,labels,context,test,label_index,class_names=load_data(); args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,values in (("text",text),("audio",audio)):
        context_values,test_values=split_without_normalization(values,context,test); np.save(args.output_dir/f"context_{name}.npy",context_values); np.save(args.output_dir/f"test_{name}.npy",test_values)
    np.save(args.output_dir/"context_labels.npy",labels[context]); np.save(args.output_dir/"test_labels.npy",labels[test]); np.save(args.output_dir/"context_indices.npy",context)
    np.savez(args.output_dir/"split_indices.npz",context=context,test=test)
    dist=lambda idx:{name:int(np.count_nonzero(labels[idx]==i)) for i,name in enumerate(class_names)}
    info={"dataset":"meld_zero","display_name":"MELD","source_dataset":"MELD","task":"3-class sentiment classification","n_classes":len(class_names),
          "class_names":class_names,"label_index":label_index,"n_context":len(context),"n_test":len(test),"modalities":["text","audio"],
          "modality_dims":{"text":text.shape[1],"audio":audio.shape[1]},"structured_modalities":[],"text_encoder":"identity","audio_encoder":"identity",
          "encoder":"MultiBench baseline utterance-level GloVe-average text + feature-selected audio","feature_variant":"baseline",
          "split":"MultiBench train+validation as Context; test as Test","context_splits":["train","validation"],"query_split":"test","seed":SEED,
          "class_distribution":{"context":dist(context),"test":dist(test)},"feature_source":str(FEATURE_DIR),
          "normalization":"none; raw encoder outputs retained without dataset-level normalization","normalization_fit_scope":"none"}
    with (args.output_dir/"info.pkl").open("wb") as file:pickle.dump(info,file)
    print(f"Saved MELD: Context={len(context)}, Test={len(test)}, dims={info['modality_dims']}")
    if args.pca48:print(project_dataset(args.output_dir.name,source_root=args.output_dir.parent,output_root=args.output_dir.parent))

if __name__=="__main__":main()

