from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    """
    Load an image from the given path and return it as a numpy array.

    Args:
        path (str | Path): Path to the image file.
    """
    return np.array(Image.open(path).convert("RGB"))
