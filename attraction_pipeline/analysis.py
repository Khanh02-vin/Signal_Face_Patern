from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from .dataset_io import read_json, write_json


NON_FEATURE_COLUMNS = {
    "video_count",
    "valid_video_count",
    "valid_video_ratio",
    "total_processed_frames",
    "avg_processed_frames",
    "quality_score",
}


@dataclass
class AnalysisResult:
    preference_vector: Dict[str, float]
    similarity_scores: List[Dict[str, float]]
    clusters: List[Dict[str, object]]
    attraction_summary: Dict[str, List[str]]


def _build_dataframe(person_profiles: List[Tuple[str, Dict[str, float]]]) -> pd.DataFrame:
    rows = []
    for person_name, profile in person_profiles:
        row = {"person": person_name}
        row.update({k: v for k, v in profile.items() if isinstance(v, (int, float))})
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)


def _describe_feature_block(pref: pd.Series, prefixes: List[str], top_n: int = 4) -> List[str]:
    selected = [col for col in pref.index if any(col.startswith(prefix) for prefix in prefixes)]
    ranked = sorted(selected, key=lambda col: abs(pref[col]), reverse=True)[:top_n]
    return [f"{name}={pref[name]:.3f}" for name in ranked]


def _select_k_by_silhouette(Xn: np.ndarray, max_k: int = 8, random_seed: int = 42) -> int:
    n_samples = len(Xn)
    if n_samples <= 2:
        return 1

    k_max = min(max_k, n_samples - 1)
    best_k = 2
    best_score = -1.0

    for k in range(2, k_max + 1):
        model = KMeans(n_clusters=k, random_state=random_seed, n_init="auto")
        labels = model.fit_predict(Xn)
        if len(set(labels)) <= 1:
            continue
        score = silhouette_score(Xn, labels)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def _cluster(
    Xn: np.ndarray,
    cluster_method: str,
    cluster_k: int,
    random_seed: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    method = cluster_method.lower().strip()
    metadata: Dict[str, float] = {}

    if method == "dbscan":
        model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        labels = model.fit_predict(Xn)
        metadata["dbscan_eps"] = float(dbscan_eps)
        metadata["dbscan_min_samples"] = float(dbscan_min_samples)
        return labels, metadata

    if method == "kmeans":
        n_clusters = max(1, min(cluster_k, len(Xn)))
        model = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init="auto")
        labels = model.fit_predict(Xn)
        metadata["selected_k"] = float(n_clusters)
        return labels, metadata

    selected_k = _select_k_by_silhouette(Xn, random_seed=random_seed)
    model = KMeans(n_clusters=selected_k, random_state=random_seed, n_init="auto")
    labels = model.fit_predict(Xn)
    metadata["selected_k"] = float(selected_k)
    metadata["k_selection"] = "silhouette"
    return labels, metadata


def _save_similarity_csv(similarity_rows: List[Dict[str, float]], csv_path: Path) -> None:
    pd.DataFrame(similarity_rows).to_csv(csv_path, index=False, encoding="utf-8")


def _save_quality_csv(df: pd.DataFrame, csv_path: Path) -> None:
    quality_cols = [
        "person",
        "quality_score",
        "valid_video_ratio",
        "valid_video_count",
        "video_count",
        "total_processed_frames",
        "avg_processed_frames",
    ]
    exists = [c for c in quality_cols if c in df.columns]
    if exists and "quality_score" in exists:
        qdf = df[exists].sort_values(by="quality_score", ascending=False)
        qdf.to_csv(csv_path, index=False, encoding="utf-8")


def _save_radar_chart(summary: Dict[str, List[str]], chart_path: Path) -> None:
    categories = ["face_structure", "eye_behavior", "expression", "emotion"]
    values = []

    for cat in categories:
        items = summary.get(cat, [])
        if not items:
            values.append(0.0)
            continue

        nums = []
        for item in items:
            try:
                nums.append(abs(float(item.split("=")[1])))
            except (IndexError, ValueError):
                continue
        values.append(float(np.mean(nums)) if nums else 0.0)

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title("Attraction Pattern Radar")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()


def _save_pca_scatter(Xn: np.ndarray, persons: List[str], labels: np.ndarray, chart_path: Path) -> None:
    if len(Xn) < 2:
        return

    pca = PCA(n_components=2, random_state=42)
    points = pca.fit_transform(Xn)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=45, alpha=0.9)
    plt.colorbar(scatter, label="Cluster")

    for idx, name in enumerate(persons):
        plt.annotate(name, (points[idx, 0], points[idx, 1]), fontsize=7, alpha=0.85)

    plt.title("PCA 2D - Person Attraction Vectors")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=170)
    plt.close()


def analyze_profiles(
    person_profile_paths: List[Path],
    output_path: Path,
    cluster_method: str = "auto_kmeans",
    cluster_k: int = 3,
    random_seed: int = 42,
    dbscan_eps: float = 1.2,
    dbscan_min_samples: int = 3,
) -> AnalysisResult:
    person_profiles = [(p.parent.name, read_json(p)) for p in person_profile_paths]
    df = _build_dataframe(person_profiles)

    feature_cols = [c for c in df.columns if c != "person" and c not in NON_FEATURE_COLUMNS]
    X = df[feature_cols].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)

    preference_vector = Xn.mean(axis=0)
    similarities = cosine_similarity(Xn, preference_vector.reshape(1, -1)).flatten()

    labels, cluster_metadata = _cluster(
        Xn,
        cluster_method=cluster_method,
        cluster_k=cluster_k,
        random_seed=random_seed,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
    )

    pref_series = pd.Series(preference_vector, index=feature_cols)
    summary = {
        "face_structure": _describe_feature_block(pref_series, ["facial_", "face_", "eye_spacing", "jaw_"]),
        "eye_behavior": _describe_feature_block(pref_series, ["eye_", "gaze_"]),
        "expression": _describe_feature_block(pref_series, ["smile_", "expression_", "head_tilt"]),
        "emotion": _describe_feature_block(pref_series, ["happy", "neutral", "surprise"]),
    }

    similarity_rows = [
        {"person": df.iloc[idx]["person"], "similarity_to_type": float(similarities[idx])}
        for idx in range(len(df))
    ]
    similarity_rows.sort(key=lambda row: row["similarity_to_type"], reverse=True)

    cluster_rows = [{"person": df.iloc[idx]["person"], "cluster": int(labels[idx])} for idx in range(len(df))]

    result = {
        "feature_columns": feature_cols,
        "preference_vector": {name: float(value) for name, value in zip(feature_cols, preference_vector)},
        "similarity_scores": similarity_rows,
        "clusters": cluster_rows,
        "cluster_method": cluster_method,
        "cluster_metadata": cluster_metadata,
        "attraction_summary": summary,
    }
    write_json(output_path, result)

    csv_path = output_path.with_name("similarity_scores.csv")
    _save_similarity_csv(similarity_rows, csv_path)

    quality_path = output_path.with_name("person_quality_scores.csv")
    _save_quality_csv(df, quality_path)

    radar_path = output_path.with_name("attraction_radar.png")
    _save_radar_chart(summary, radar_path)

    pca_path = output_path.with_name("pca_person_vectors.png")
    _save_pca_scatter(Xn, df["person"].tolist(), labels, pca_path)

    return AnalysisResult(
        preference_vector=result["preference_vector"],
        similarity_scores=similarity_rows,
        clusters=cluster_rows,
        attraction_summary=summary,
    )