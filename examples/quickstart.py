"""Minimal synthetic multimodal quick-start example for Fusion Anything (FAM)."""

from __future__ import annotations

import numpy as np

from fusion_anything import FusionAnythingClassifier


def main() -> None:
    rng = np.random.default_rng(42)

    # Synthetic labeled Context (100 samples, 3 classes) and unlabeled Query (20 samples)
    n_context, n_query, n_classes = 100, 20, 3
    context = {
        "image": rng.standard_normal((n_context, 512), dtype=np.float32),
        "tabular": rng.standard_normal((n_context, 12), dtype=np.float32),
    }
    labels = rng.integers(0, n_classes, size=n_context, dtype=np.int64)
    query = {
        "image": rng.standard_normal((n_query, 512), dtype=np.float32),
        "tabular": rng.standard_normal((n_query, 12), dtype=np.float32),
    }

    # preprocessing="pca" projects unstructured modalities (>48 dims) using Context-only PCA.
    # Declare structured/tabular modalities so their columns are preserved directly.
    model = FusionAnythingClassifier(device="cuda:0", preprocessing="pca")
    model.fit(context, labels, structured_modalities=["tabular"])

    probabilities = model.predict_proba(query)
    predictions = model.classes_[probabilities.argmax(axis=1)]
    print("Probabilities shape:", probabilities.shape)
    print("Predicted classes:  ", predictions)


if __name__ == "__main__":
    main()
