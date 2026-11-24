"""Public NumPy interface for fixed-weight, context-conditioned classification."""
from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from fusion_anything.data import GroupData


class FusionAnythingClassifier:
    """Set a labeled Context with fit(), then predict unlabeled feature arrays.

    fit does not train neural weights. preprocessing='pca' fits Context-only
    PCA on unstructured modalities; 'none' accepts already prepared features.
    CUDA is required. Model loading is lazy, on the first nonempty prediction.

    Args:
        checkpoint: Optional path to the supported full-state checkpoint.
            Its content must match the pinned model version.
        device: Visible CUDA device, such as "cuda:0".
        preprocessing: "none" for prepared features or "pca" for Context-only
            projection of unstructured features.
        chunk_size: Maximum Query rows per inference chunk. Every chunk uses
            all Context rows; CUDA memory pressure may reduce the chunk size.

    Attributes:
        classes_: Integer class IDs in probability-column order, set by fit().
        modality_names_: Modality order established by fit().
        runtime_: Measurements from the most recent nonempty prediction.

    Notes:
        Only this class and its documented methods are supported application
        interfaces. Underscore-prefixed members are internal conventions,
        not access controls or a mechanism for hiding source code.
    """

    def __init__(
        self, checkpoint: str | Path | None = None, *, device: str = "cuda:0",
        preprocessing: str = "none", chunk_size: int = 128,
    ) -> None:
        if preprocessing not in {"none", "pca"}:
            raise ValueError("preprocessing must be 'none' or 'pca'")
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        self.checkpoint = checkpoint
        self.device = device
        self.preprocessing = preprocessing
        self.chunk_size = chunk_size
        self._wrapper = None

    @staticmethod
    def _arrays(features: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not isinstance(features, Mapping) or not 1 <= len(features) <= 8:
            raise ValueError("features must map 1 to 8 modality names to arrays")
        output = {}
        rows = None
        for name, values in features.items():
            if not isinstance(name, str) or not name:
                raise ValueError("modality names must be nonempty strings")
            values = np.array(values, dtype=np.float32, copy=True)
            if values.ndim != 2 or values.shape[1] == 0:
                raise ValueError(f"{name}: expected [samples, features] with nonzero width")
            if not np.isfinite(values).all():
                raise ValueError(f"{name}: features contain NaN or Inf")
            if rows is not None and len(values) != rows:
                raise ValueError(f"{name}: sample count differs from other modalities")
            rows = len(values)
            output[name] = values
        return output

    def fit(
        self, features: Mapping[str, np.ndarray], labels: np.ndarray, *,
        structured_modalities: Sequence[str] = (),
        context_indices: np.ndarray | None = None,
    ) -> FusionAnythingClassifier:
        """Store Context and preprocessing; leave checkpoint weights unchanged.

        Args:
            features: Mapping of modality names to finite 2D feature arrays.
                Modalities must have the same nonzero number of Context rows.
            labels: One-dimensional integer labels, continuously encoded from
                0. Context must contain all 2 to 10 supported classes.
            structured_modalities: Modalities whose feature columns are kept
                unchanged rather than projected by PCA.
            context_indices: Unique original row indices used to reproduce
                canonical PCA fitting order; does not reorder the stored rows.

        Returns:
            This classifier with the Context and preprocessing state replaced.

        Raises:
            ValueError: Features, labels, modality declarations, or original
                row indices violate the input contract.
        """
        arrays = self._arrays(features)
        n = len(next(iter(arrays.values())))
        y = np.asarray(labels)
        if n == 0 or y.ndim != 1 or len(y) != n or not np.issubdtype(y.dtype, np.integer):
            raise ValueError("Context requires nonempty arrays and matching 1D integer labels")
        classes = np.unique(y)
        if not 2 <= len(classes) <= 10 or not np.array_equal(classes, np.arange(len(classes))):
            raise ValueError("Context must contain 2 to 10 classes encoded continuously from 0")
        structured = set(structured_modalities)
        if not structured.issubset(arrays):
            raise ValueError("structured_modalities contains unknown modality names")
        if context_indices is None:
            order = np.arange(n)
        else:
            indices = np.asarray(context_indices)
            if (indices.shape != (n,) or not np.issubdtype(indices.dtype, np.integer)
                    or len(np.unique(indices)) != n):
                raise ValueError("context_indices must contain one unique integer per Context row")
            order = np.argsort(indices, kind="stable")
        models = {}
        dims = {name: values.shape[1] for name, values in arrays.items()}
        for name, values in arrays.items():
            if name in structured:
                continue
            if self.preprocessing == "none":
                if values.shape[1] > 48:
                    raise ValueError(f"{name}: unstructured features exceed 48 columns; use preprocessing='pca'")
            else:
                components = min(48, n, values.shape[1])
                pca = PCA(n_components=components, svd_solver="auto", random_state=42)
                canonical = pca.fit_transform(np.ascontiguousarray(values[order])).astype(np.float32)
                projected = np.zeros((n, 48), dtype=np.float32)
                projected[order, :components] = canonical
                arrays[name] = projected
                models[name] = pca
        self.modality_names_ = tuple(arrays)
        self.structured_modalities_ = structured
        self.classes_ = classes.astype(np.int64)
        self._input_dims = dims
        self._pca = models
        self._context = arrays
        self._labels = np.array(y, dtype=np.int64, copy=True)
        return self

    def _load_model(self) -> None:
        import torch
        from fusion_anything.runner import (
            DEFAULT_TABPFN_ROOT, LORA_SHA256, build_model, resolve_checkpoint_path,
        )
        device = torch.device(self.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Fusion Anything inference requires an available CUDA device")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        args = Namespace(
            tabpfn_root=DEFAULT_TABPFN_ROOT,
            lora_checkpoint=resolve_checkpoint_path(self.checkpoint),
            lora_sha256=LORA_SHA256,
        )
        self._wrapper, self.checkpoint_metadata_ = build_model(args, device)

    def predict_proba(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        """Predict probabilities without requiring Query labels.

        Args:
            features: Query arrays with the same modality names and original
                feature widths as Context. Modality dictionary order may differ.

        Returns:
            Float64 probabilities shaped [query_rows, number_of_classes], with
            columns in classes_ order. Empty Query returns an empty matrix.

        Raises:
            ValueError: Query features violate the Context feature contract.
            RuntimeError: fit() has not been called, CUDA is unavailable,
                model assets fail validation, or probabilities are not finite.
        """
        if not hasattr(self, "_context"):
            raise RuntimeError("Call fit() with labeled Context before prediction")
        arrays = self._arrays(features)
        if set(arrays) != set(self.modality_names_):
            raise ValueError("Prediction modalities must match Context modalities")
        arrays = {name: arrays[name] for name in self.modality_names_}
        for name, values in arrays.items():
            if values.shape[1] != self._input_dims[name]:
                raise ValueError(f"{name}: feature width differs from Context")
            if name in self._pca:
                pca = self._pca[name]
                projected = np.zeros((len(values), 48), dtype=np.float32)
                if len(values):
                    projected[:, :pca.n_components_] = pca.transform(values).astype(np.float32)
                arrays[name] = projected
        n_test = len(next(iter(arrays.values())))
        if n_test == 0:
            return np.empty((0, len(self.classes_)), dtype=np.float64)
        if self._wrapper is None:
            self._load_model()
        # The shared engine supplies zero placeholders for wrapper loss only.
        # Target embeddings consume only the labeled Context prefix.
        blocks = []
        start = 0
        for name in self.modality_names_:
            end = start + self._context[name].shape[1]
            blocks.append({"name": name, "column_start": start, "column_end_exclusive": end})
            start = end
        data = GroupData(
            group_code="API", group_name="API", data_dir=Path("."),
            context_x=np.concatenate(list(self._context.values()), axis=1),
            context_y=self._labels,
            test_x=np.concatenate(list(arrays.values()), axis=1),
            test_y=np.zeros(n_test, dtype=np.int64),
            feature_blocks=tuple(blocks), test_feature_blocks=tuple(blocks), info={},
        )
        args = getattr(self, "_inference_args", None) or Namespace(
            device=self.device, dtype="float32", finetuned_chunk_size=self.chunk_size,
            enforce_query_chunk_invariance=False, invariance_atol=5e-4,
        )
        from fusion_anything.inference import _fine_tuned_probabilities
        types = ["structured" if name in self.structured_modalities_ else "unstructured"
                 for name in self.modality_names_]
        probabilities, self.runtime_ = _fine_tuned_probabilities(
            wrapper=self._wrapper, data=data, types=types, args=args,
        )
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Model returned non-finite probabilities")
        return probabilities

    def predict(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        """Return one integer class ID per Query row.

        Calls predict_proba(), so invoking both methods runs inference twice.
        If probabilities are already available, use classes_[probabilities.argmax(1)].
        """
        probabilities = self.predict_proba(features)
        return self.classes_[probabilities.argmax(axis=1)]
