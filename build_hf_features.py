"""
Build public-safe features dataset from processed output.
Extracts numerical features per video → aggregates per person.
Safe to public on HuggingFace (no raw faces).
"""
import json
import csv
import hashlib
from pathlib import Path
from collections import defaultdict

FEATURES_DIR = Path("output")       # your processed JSON files
METADATA_CSV = Path("metadata.csv") # your existing metadata
EMBEDDINGS_DIR = Path("embeddings")
FEATURES_CSV = Path("embeddings/person_features.csv")
METADATA_PUB_CSV = Path("embeddings/metadata.csv")

# Feature columns we want to keep (numerical only)
FEATURE_KEYS = [
    "facial_symmetry_mean", "facial_symmetry_std", "facial_symmetry_median",
    "face_width_height_ratio_mean", "face_width_height_ratio_std",
    "eye_spacing_ratio_mean", "eye_spacing_ratio_std",
    "jaw_ratio_mean", "jaw_ratio_std",
    "smile_intensity_mean", "smile_intensity_std", "smile_intensity_max",
    "eye_contact_mean", "eye_contact_std",
    "eye_openness_mean", "eye_openness_std",
    "gaze_stability_mean", "gaze_stability_std",
    "head_tilt_mean", "head_tilt_std",
    "expression_speed_mean", "expression_speed_std",
    "happy_mean", "neutral_mean", "surprise_mean",
    "processed_frames",
]

def slugify_name(name: str) -> str:
    """Normalize folder/person name to slug."""
    import re
    name = name.strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name or "unknown"


def extract_video_features(video_json: Path) -> dict | None:
    """Load a single video's feature JSON."""
    try:
        with video_json.open(encoding="utf-8") as f:
            data = json.load(f)
        return {k: data.get(k, 0.0) for k in FEATURE_KEYS}
    except Exception as e:
        print(f"  [!] Skipping {video_json.name}: {e}")
        return None


def build_person_features(features_dir: Path) -> dict[str, list[dict]]:
    """Aggregate features per person folder."""
    person_videos = defaultdict(list)

    for person_dir in features_dir.iterdir():
        if not person_dir.is_dir():
            continue
        person_id = slugify_name(person_dir.name)

        for video_json in sorted(person_dir.glob("*.json")):
            feat = extract_video_features(video_json)
            if feat:
                feat["video_name"] = video_json.stem
                feat["video_file"] = f"raw/{person_id}/{video_json.stem}.mp4"
                person_videos[person_id].append(feat)

    return person_videos


def aggregate_person_stats(person_videos: dict[str, list[dict]]) -> list[dict]:
    """Aggregate per-video stats into per-person stats (mean across videos)."""
    rows = []

    for person_id, videos in sorted(person_videos.items()):
        if not videos:
            continue

        row = {"person_id": person_id, "num_videos": len(videos)}

        for feat_name in FEATURE_KEYS:
            values = [v[feat_name] for v in videos if feat_name in v]
            if values:
                row[f"{feat_name}_mean"] = sum(values) / len(values)
                row[f"{feat_name}_std"] = (
                    (sum((v - row[f"{feat_name}_mean"])**2 for v in values) / len(values)) ** 0.5
                    if len(values) > 1 else 0.0
                )
            else:
                row[f"{feat_name}_mean"] = 0.0
                row[f"{feat_name}_std"] = 0.0

        rows.append(row)

    return rows


def load_original_metadata(meta_csv: Path) -> dict[str, dict]:
    """Load original metadata to get person_name mapping."""
    if not meta_csv.exists():
        return {}
    mapping = {}
    with meta_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            mapping[row["person_id"]] = row.get("person_name", row["person_id"])
    return mapping


def main():
    print("[1] Scanning processed features...")
    person_videos = build_person_features(FEATURES_DIR)
    print(f"   Found {len(person_videos)} persons")

    print("[2] Aggregating per-person statistics...")
    aggregated = aggregate_person_stats(person_videos)

    name_map = load_original_metadata(METADATA_CSV)

    # Ensure embeddings dir
    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    # 1) Write per-person aggregated features CSV
    if aggregated:
        fieldnames = list(aggregated[0].keys())
        # add person_name at the beginning
        fieldnames.insert(0, "person_name")
        with FEATURES_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in aggregated:
                row["person_name"] = name_map.get(row["person_id"], row["person_id"])
                w.writerow(row)
        print(f"   OK: Saved person features -> {FEATURES_CSV} ({len(aggregated)} rows)")

    # 2) Write per-video features CSV (flattened)
    video_rows = []
    for person_id, videos in sorted(person_videos.items()):
        for v in videos:
            video_rows.append({"person_id": person_id, "person_name": name_map.get(person_id, person_id), **v})

    if video_rows:
        vid_csv = EMBEDDINGS_DIR / "video_features.csv"
        fieldnames = list(video_rows[0].keys())
        with vid_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(video_rows)
        print(f"   OK Saved video features → {vid_csv} ({len(video_rows)} rows)")

    # 3) Build public metadata CSV (no raw video paths, no sha256)
    if METADATA_CSV.exists():
        with METADATA_CSV.open(encoding="utf-8", newline="") as f:
            orig_rows = list(csv.DictReader(f))

        pub_rows = []
        for row in orig_rows:
            pub_rows.append({
                "sample_id": row["sample_id"],
                "person_id": row["person_id"],
                "person_name": row["person_name"],
                "split": row.get("split", "train"),
                "feature_path": f"embeddings/video_features.csv",  # placeholder, actual row added below
            })

        # Write one row per unique person for simplicity
        seen = set()
        person_pub_rows = []
        for row in orig_rows:
            pid = row["person_id"]
            if pid in seen:
                continue
            seen.add(pid)
            person_pub_rows.append({
                "person_id": pid,
                "person_name": row["person_name"],
                "num_samples": sum(1 for r in orig_rows if r["person_id"] == pid),
                "split": row.get("split", "train"),
            })

        with METADATA_PUB_CSV.open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(person_pub_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(person_pub_rows)
        print(f"   OK Saved public metadata → {METADATA_PUB_CSV} ({len(person_pub_rows)} persons)")

    print("\n[DONE] Next steps:")
    print("   1. Review embeddings/person_features.csv")
    print("   2. git add embeddings/")
    print("   3. git commit -m 'Add public-safe person/video features'")
    print("   4. git push origin main")


if __name__ == "__main__":
    main()
