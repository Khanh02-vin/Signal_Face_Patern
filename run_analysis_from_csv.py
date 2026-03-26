#!/usr/bin/env python3
"""
Generate analysis outputs (radar chart, PCA scatter, CSV reports)
from already-processed Signal_Face CSV data.
No raw videos needed — runs analysis only.

Usage:
    python run_analysis_from_csv.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 1. Load processed CSV data ──────────────────────────────────────────────
print("[1/4] Loading Signal_Face/ CSVs...")
pf = pd.read_csv("Signal_Face/person_features.csv")
vf = pd.read_csv("Signal_Face/video_features.csv")
meta = pd.read_csv("Signal_Face/metadata.csv")

print(f"    person_features : {len(pf)} rows")
print(f"    video_features  : {len(vf)} rows")
print(f"    metadata        : {len(meta)} rows")

# ── 2. Build feature matrix ──────────────────────────────────────────────────
print("[2/4] Building feature matrix...")

non_feature = {"person_name", "person_id", "person", "video_count", "valid_video_count",
                "valid_video_ratio", "total_processed_frames", "avg_processed_frames",
                "quality_score", "num_videos", "num_samples", "split",
                # video_features extra cols
                "video_name", "video_file", "processed_frames",
                # person_features extra cols
                "processed_frames_mean", "processed_frames_std",
                "facial_symmetry_mean_std", "facial_symmetry_median_std",
                "facial_symmetry_median_mean",
                "face_width_height_ratio_mean_std", "face_width_height_ratio_std_mean",
                "face_width_height_ratio_std_std",
                "eye_spacing_ratio_mean_std", "eye_spacing_ratio_std_mean", "eye_spacing_ratio_std_std",
                "jaw_ratio_mean_std", "jaw_ratio_std_mean", "jaw_ratio_std_std",
                "smile_intensity_mean_std", "smile_intensity_std_mean", "smile_intensity_std_std", "smile_intensity_max_std",
                "eye_contact_mean_std", "eye_contact_std_mean", "eye_contact_std_std",
                "eye_openness_mean_std", "eye_openness_std_mean", "eye_openness_std_std",
                "gaze_stability_mean_std", "gaze_stability_std_mean", "gaze_stability_std_std",
                "head_tilt_mean_std", "head_tilt_std_mean", "head_tilt_std_std",
                "expression_speed_mean_std", "expression_speed_std_mean", "expression_speed_std_std",
                "happy_mean_std", "neutral_mean_std", "surprise_mean_std",
                # extra non-feature
                "facial_symmetry_std_mean", "facial_symmetry_std_std",
            }

# Aggregate video_features per person → person-level
vf_person = (
    vf.drop(columns=[c for c in ["video_name","video_file","processed_frames","person_name"]
                     if c in vf.columns], errors="ignore")
    .groupby("person_id", as_index=False)
    .mean()
)

# Merge with person features
df = pf.merge(vf_person, on="person_id", how="left", suffixes=("", "_vf"))

feature_cols = [c for c in df.columns
                if c not in non_feature
                and not c.endswith("_vf")
                and not c.endswith("_x")
                and not c.endswith("_y")
                and df[c].dtype in ["float64", "int64"]]
feature_cols = sorted(set(feature_cols))
print(f"    Feature columns: {len(feature_cols)}")

persons = df["person_name"].tolist()
X = df[feature_cols].fillna(0).to_numpy(dtype=np.float32)

scaler = StandardScaler()
Xn = scaler.fit_transform(X)
print(f"    Feature matrix: {Xn.shape}")

# ── 3. Clustering + Preference Vector ───────────────────────────────────────
print("[3/4] Clustering + cosine similarity...")

# Auto-select k via silhouette
best_k, best_score = 2, -1.0
for k in range(2, min(8, len(Xn))):
    labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(Xn)
    if len(set(labels)) > 1:
        score = silhouette_score(Xn, labels)
        if score > best_score:
            best_score, best_k = score, k

labels = KMeans(n_clusters=best_k, random_state=42, n_init="auto").fit_predict(Xn)
print(f"    Selected k={best_k} (silhouette={best_score:.4f})")

pref_vector = Xn.mean(axis=0)
similarities = cosine_similarity(Xn, pref_vector.reshape(1, -1)).flatten()

# ── 4. Save outputs ──────────────────────────────────────────────────────────
print("[4/4] Saving outputs...")

# 4a. similarity_scores.csv
sim_df = pd.DataFrame({
    "person": persons,
    "person_id": df["person_id"].tolist(),
    "similarity_to_type": similarities,
    "cluster": labels,
    "num_videos": df["num_videos"].tolist() if "num_videos" in df.columns
                  else df["num_samples"].tolist() if "num_samples" in df.columns else 0,
}).sort_values("similarity_to_type", ascending=False)
sim_df.to_csv(OUTPUT_DIR / "similarity_scores.csv", index=False, encoding="utf-8")
print(f"    [OK] similarity_scores.csv")

# 4b. attraction_report.json
import json
report = {
    "num_persons": int(len(df)),
    "num_videos": int(len(vf)),
    "num_features": len(feature_cols),
    "feature_columns": feature_cols,
    "cluster_method": "auto_kmeans",
    "selected_k": int(best_k),
    "silhouette_score": float(best_score),
    "similarity_scores": sim_df.to_dict(orient="records"),
    "cluster_labels": {p: int(l) for p, l in zip(persons, labels)},
}
with open(OUTPUT_DIR / "attraction_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"    [OK] attraction_report.json")

# 4c. Radar chart
categories = ["Face Structure", "Eye Behavior", "Expression", "Emotion"]

# Aggregate top features per category
def cat_score(prefix):
    cols = [c for c in feature_cols if c.startswith(prefix)]
    if not cols:
        return 0.0
    return float(np.mean([abs(pref_vector[feature_cols.index(c)]) for c in cols]))

radar_values = [
    cat_score("facial_symmetry") + cat_score("face_width") + cat_score("eye_spacing") + cat_score("jaw"),
    cat_score("eye_contact") + cat_score("eye_open") + cat_score("gaze"),
    cat_score("smile") + cat_score("head_tilt") + cat_score("expression"),
    cat_score("happy") + cat_score("neutral") + cat_score("surprise"),
]
# Normalize to 0-1 for radar
max_val = max(radar_values) if max(radar_values) > 0 else 1
radar_values = [v / max_val for v in radar_values]

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
rv = radar_values + radar_values[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
ax.plot(angles, rv, linewidth=2.5, color="#4f46e5")
ax.fill(angles, rv, alpha=0.25, color="#4f46e5")
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=13)
ax.set_ylim(0, 1.05)
ax.set_title("Facial Signal Profile — Radar", fontsize=15, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "attraction_radar.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"    [OK] attraction_radar.png")

# 4d. PCA scatter
pca = PCA(n_components=2, random_state=42)
points = pca.fit_transform(Xn)
cluster_colors = ["#4f46e5", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]

fig, ax = plt.subplots(figsize=(10, 8))
for cluster_id in sorted(set(labels)):
    mask = labels == cluster_id
    ax.scatter(
        points[mask, 0], points[mask, 1],
        c=cluster_colors[cluster_id % len(cluster_colors)],
        label=f"Cluster {cluster_id} ({sum(mask)} people)",
        s=70, alpha=0.85,
    )
    for idx in np.where(mask)[0]:
        ax.annotate(
            persons[idx], (points[idx, 0], points[idx, 1]),
            fontsize=7.5, alpha=0.8,
            xytext=(4, 4), textcoords="offset points",
        )

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
ax.set_title("PCA 2D — Facial Signal Vectors", fontsize=15, fontweight="bold")
ax.legend(loc="best", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pca_person_vectors.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"    [OK] pca_person_vectors.png")

# 4e. person_quality_scores.csv
if "quality_score" in pf.columns:
    q_df = pf[["person_name", "num_videos", "quality_score",
                "processed_frames_mean", "facial_symmetry_mean_mean",
                "smile_intensity_mean_mean"]].copy()
    q_df = q_df.sort_values("quality_score", ascending=False)
    q_df.to_csv(OUTPUT_DIR / "person_quality_scores.csv", index=False, encoding="utf-8")
    print(f"    [OK] person_quality_scores.csv")

print("\n[OK] All outputs saved to output/")
print(f"   attraction_report.json, similarity_scores.csv, person_quality_scores.csv")
print(f"   attraction_radar.png, pca_person_vectors.png")
