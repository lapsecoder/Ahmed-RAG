"""A single embedding, tied to the chunk it was produced from."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Embedding:
    """A float32 vector plus the id of the chunk it describes.

    This type is an *indexing-time* helper. It must never be stored in the
    FAISS position -> metadata mapping; that mapping holds ``DocumentChunk``.
    """

    vector: NDArray[np.float32]
    chunk_id: str = field(default="")
