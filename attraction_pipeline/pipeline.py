from __future__ import annotations

import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from .analysis import analyze_profiles
from .config import PipelineConfig
from .dataset_io import aggregate_video_stats, iter_person_dirs, iter_videos, write_json
from .video_features import VideoFeatureExtractor


def _process_person_worker(
    person_dir_str: str,
    output_root_str: str,
    frame_step: int,
    max_frames_per_video: int | None,
    min_face_confidence: float,
) -> Tuple[Path | None, Dict]:
    person_dir = Path(person_dir_str)
    output_root = Path(output_root_str)

    extractor = VideoFeatureExtractor(min_face_confidence=min_face_confidence)
    video_stats = []
    videos = list(iter_videos(person_dir))

    for video_path in videos:
        json_path = output_root / person_dir.name / f"{video_path.stem}.json"
        try:
            stats = extractor.process_video(
                str(video_path),
                frame_step=frame_step,
                max_frames=max_frames_per_video,
            )
            if not stats:
                continue

            stats["video_name"] = video_path.name
            write_json(json_path, stats)
            video_stats.append(stats)
        except Exception:
            traceback.print_exc()

    person_profile, quality = aggregate_video_stats(video_stats, total_videos=len(videos))
    if person_profile:
        person_profile["person_name"] = person_dir.name
        profile_path = output_root / person_dir.name / "person_profile.json"
        write_json(profile_path, person_profile)
        return profile_path, quality

    return None, quality


def run_pipeline(config: PipelineConfig) -> None:
    """Run end-to-end pipeline: videos -> features -> person profiles -> attraction report."""
    dataset_root = config.dataset_root
    output_root = config.resolved_output_root()
    person_dirs = list(iter_person_dirs(dataset_root))
    person_profile_paths: List[Path] = []

    if config.use_multiprocessing:
        print("[INFO] Multiprocessing enabled.")
        with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
            futures = [
                executor.submit(
                    _process_person_worker,
                    str(person_dir),
                    str(output_root),
                    config.frame_step,
                    config.max_frames_per_video,
                    config.min_face_confidence,
                )
                for person_dir in person_dirs
            ]
            for future in as_completed(futures):
                profile_path, _quality = future.result()
                if profile_path is not None:
                    person_profile_paths.append(profile_path)
    else:
        extractor = VideoFeatureExtractor(min_face_confidence=config.min_face_confidence)

        for person_dir in person_dirs:
            print(f"[INFO] Processing person: {person_dir.name}")
            video_stats = []
            videos = list(iter_videos(person_dir))

            for video_path in videos:
                json_path = output_root / person_dir.name / f"{video_path.stem}.json"
                try:
                    stats = extractor.process_video(
                        str(video_path),
                        frame_step=config.frame_step,
                        max_frames=config.max_frames_per_video,
                    )
                    if not stats:
                        print(f"[WARN] No valid face frames in {video_path.name}")
                        continue

                    stats["video_name"] = video_path.name
                    write_json(json_path, stats)
                    video_stats.append(stats)
                    print(f"[OK] Saved video stats: {json_path}")
                except Exception as error:
                    print(f"[ERROR] Failed processing {video_path}: {error}")
                    traceback.print_exc()

            person_profile, _quality = aggregate_video_stats(video_stats, total_videos=len(videos))
            if person_profile:
                person_profile["person_name"] = person_dir.name
                profile_path = output_root / person_dir.name / "person_profile.json"
                write_json(profile_path, person_profile)
                person_profile_paths.append(profile_path)
                print(f"[OK] Saved person profile: {profile_path}")

    if not person_profile_paths:
        print("[WARN] No person profiles generated. Stopping before analysis.")
        return

    report_path = output_root / "attraction_report.json"
    analyze_profiles(
        person_profile_paths=person_profile_paths,
        output_path=report_path,
        cluster_method=config.cluster_method,
        cluster_k=config.cluster_k,
        random_seed=config.random_seed,
        dbscan_eps=config.dbscan_eps,
        dbscan_min_samples=config.dbscan_min_samples,
    )
    print(f"[DONE] Attraction report created: {report_path}")