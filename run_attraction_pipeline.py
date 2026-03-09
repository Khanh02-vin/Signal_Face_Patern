from pathlib import Path

from attraction_pipeline.config import PipelineConfig
from attraction_pipeline.pipeline import run_pipeline


if __name__ == "__main__":
    # Update these paths to your environment if needed.
    dataset_root = Path(r"D:\AI-ML\New folder\Signal_Face_Patern\raw")
    output_root = Path(r"D:\AI-ML\New folder\Signal_Face_Patern\output")

    config = PipelineConfig(
        dataset_root=dataset_root,
        output_root=output_root,
        frame_step=5,
        min_face_confidence=0.5,
        max_frames_per_video=None,
        cluster_k=3,
        cluster_method="auto_kmeans",  # auto_kmeans | kmeans | dbscan
        dbscan_eps=1.2,
        dbscan_min_samples=3,
        use_multiprocessing=False,      # True to enable multiprocessing
        max_workers=None,               # e.g. 4
    )

    run_pipeline(config)