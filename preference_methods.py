from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path("output")
NON_FEATURE_COLUMNS = {
    "person_name",
    "video_count",
    "valid_video_count",
    "valid_video_ratio",
    "total_processed_frames",
    "avg_processed_frames",
    "quality_score",
}


def load_person_profiles(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict] = []
    quality_rows: List[Dict] = []

    for person_dir in sorted(output_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        profile_path = person_dir / "person_profile.json"
        if not profile_path.exists():
            continue

        with profile_path.open("r", encoding="utf-8") as file:
            profile = json.load(file)

        row = {"person": person_dir.name}
        for key, value in profile.items():
            if isinstance(value, (int, float)) and key not in NON_FEATURE_COLUMNS:
                row[key] = float(value)
        rows.append(row)

        quality_rows.append(
            {
                "person": person_dir.name,
                "quality_score": float(profile.get("quality_score", 0.5)),
                "valid_video_ratio": float(profile.get("valid_video_ratio", 0.5)),
                "avg_processed_frames": float(profile.get("avg_processed_frames", 0.0)),
            }
        )

    feature_df = pd.DataFrame(rows).fillna(0.0)
    quality_df = pd.DataFrame(quality_rows).fillna(0.0)
    return feature_df, quality_df


def compute_similarity(x_norm: np.ndarray, pref_vec: np.ndarray) -> np.ndarray:
    return cosine_similarity(x_norm, pref_vec.reshape(1, -1)).flatten()


def mean_all(x_norm: np.ndarray) -> np.ndarray:
    return x_norm.mean(axis=0)


def weighted_mean(x_norm: np.ndarray, quality_df: pd.DataFrame) -> np.ndarray:
    q = quality_df["quality_score"].to_numpy(dtype=np.float32)
    v = quality_df["valid_video_ratio"].to_numpy(dtype=np.float32)
    f = quality_df["avg_processed_frames"].to_numpy(dtype=np.float32)

    if np.max(f) > 0:
        f = f / np.max(f)

    weights = 0.6 * q + 0.3 * v + 0.1 * f
    weights = np.clip(weights, 1e-6, None)
    weights = weights / np.sum(weights)
    return np.average(x_norm, axis=0, weights=weights)


def core_percentile(x_norm: np.ndarray, keep_ratio: float = 0.4) -> np.ndarray:
    centroid = x_norm.mean(axis=0)
    dists = np.linalg.norm(x_norm - centroid, axis=1)
    keep_n = max(3, int(len(dists) * keep_ratio))
    keep_idx = np.argsort(dists)[:keep_n]
    return x_norm[keep_idx].mean(axis=0)


def largest_cluster_center(x_norm: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    n_clusters = max(2, min(n_clusters, len(x_norm)))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = model.fit_predict(x_norm)

    counts = pd.Series(labels).value_counts()
    dominant_label = int(counts.index[0])
    idx = np.where(labels == dominant_label)[0]
    return x_norm[idx].mean(axis=0)


def main() -> None:
    feature_df, quality_df = load_person_profiles(OUTPUT_DIR)
    if feature_df.empty:
        print("[WARN] No person_profile.json found in output.")
        return

    feature_cols = [c for c in feature_df.columns if c != "person"]
    x = feature_df[feature_cols].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    x_norm = scaler.fit_transform(x)

    methods = {
        "mean_all": mean_all(x_norm),
        "weighted_mean": weighted_mean(x_norm, quality_df),
        "core_percentile_40": core_percentile(x_norm, keep_ratio=0.4),
        "largest_cluster_center": largest_cluster_center(x_norm, n_clusters=3),
    }

    report: Dict[str, Dict] = {}
    sim_df = pd.DataFrame({"person": feature_df["person"]})

    for method_name, pref_vec in methods.items():
        sims = compute_similarity(x_norm, pref_vec)
        sim_df[method_name] = sims

        top_idx = np.argsort(sims)[::-1][:10]
        top_people = [
            {"person": feature_df.iloc[i]["person"], "similarity": float(sims[i])}
            for i in top_idx
        ]

        top_feat_idx = np.argsort(np.abs(pref_vec))[::-1][:12]
        top_features = [
            {"feature": feature_cols[i], "value": float(pref_vec[i])}
            for i in top_feat_idx
        ]

        report[method_name] = {
            "top_people": top_people,
            "top_features": top_features,
            "preference_vector": {feature_cols[i]: float(pref_vec[i]) for i in range(len(feature_cols))},
        }

    # default core vector for downstream (v2 default = core_percentile_40)
    core_vec = methods["core_percentile_40"]
    core_json = {
        "method": "core_percentile_40",
        "vector": {feature_cols[i]: float(core_vec[i]) for i in range(len(feature_cols))},
    }

    with (OUTPUT_DIR / "preference_strategies_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with (OUTPUT_DIR / "core_preference_vector.json").open("w", encoding="utf-8") as f:
        json.dump(core_json, f, ensure_ascii=False, indent=2)

    sim_df.to_csv(OUTPUT_DIR / "preference_strategy_similarity_scores.csv", index=False, encoding="utf-8")

    print("[DONE] output/preference_strategies_report.json")
    print("[DONE] output/core_preference_vector.json")
    print("[DONE] output/preference_strategy_similarity_scores.csv")


if __name__ == "__main__":
    main()