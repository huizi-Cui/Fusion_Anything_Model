"""Set the dataset and encoder roots here, or use the corresponding environment variables."""

import os
from pathlib import Path


def dataset_path(relative: str) -> Path:
    # Preserve the relative layout specified at the top of each dataset script.
    return Path(os.environ.get("FUSION_DATA_ROOT", "/path/to/dataset_bundle")).expanduser() / relative


def encoder_path(relative: str) -> Path:
    return Path(os.environ.get("FUSION_ENCODER_ROOT", "/path/to/encoder_weights")).expanduser() / relative
