from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def iter_person_dirs(dataset_root: Path) -> Iterable[Path]:
    for person_dir in sorted(dataset_root.iterdir()):
        if person_dir.is_dir():
            yield person_dir


def iter_videos(person_dir: Path) -> Iterable[Path]:
    valid_ext = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    for file_path in sorted(person_dir.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in valid_ext:
            yield file_path


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def aggregate_video_stats(
    video_feature_dicts: List[Dict[str, float]],
    total_videos: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Aggregate video stats and build quality metadata for a person."""
    if not video_feature_dicts:
        quality = {
            "video_count": float(total_videos),
            "valid_video_count": 0.0,
            "valid_video_ratio": 0.0,
            "total_processed_frames": 0.0,
            "avg_processed_frames": 0.0,
            "quality_score": 0.0,
        }
        return {}, quality

    keys = sorted(set().union(*[d.keys() for d in video_feature_dicts]))
    profile: Dict[str, float] = {}

    for key in keys:
        values = [float(d[key]) for d in video_feature_dicts if key in d and isinstance(d[key], (int, float))]
        if values:
            profile[key] = float(sum(values) / len(values))

    valid_count = len(video_feature_dicts)
    processed_frames = [float(d.get("processed_frames", 0.0)) for d in video_feature_dicts]
    total_frames = float(sum(processed_frames))
    avg_frames = float(total_frames / valid_count) if valid_count else 0.0
    valid_ratio = float(valid_count / total_videos) if total_videos > 0 else 0.0

    frame_factor = min(avg_frames / 150.0, 1.0)
    quality_score = 0.6 * valid_ratio + 0.4 * frame_factor

    quality = {
        "video_count": float(total_videos),
        "valid_video_count": float(valid_count),
        "valid_video_ratio": valid_ratio,
        "total_processed_frames": total_frames,
        "avg_processed_frames": avg_frames,
        "quality_score": float(quality_score),
    }

    profile.update(quality)
    return profile, quality