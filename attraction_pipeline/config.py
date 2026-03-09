from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    """Configuration for the attraction discovery pipeline."""

    dataset_root: Path
    output_root: Optional[Path] = None

    frame_step: int = 5
    min_face_confidence: float = 0.5
    max_frames_per_video: Optional[int] = None
    random_seed: int = 42

    cluster_k: int = 3
    cluster_method: str = "auto_kmeans"  # auto_kmeans | kmeans | dbscan
    dbscan_eps: float = 1.2
    dbscan_min_samples: int = 3

    use_multiprocessing: bool = False
    max_workers: Optional[int] = None

    def resolved_output_root(self) -> Path:
        """Use dataset root if output root is not explicitly provided."""
        return self.output_root if self.output_root else self.dataset_root