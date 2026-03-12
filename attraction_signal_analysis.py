from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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


def load_person_feature_df(output_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []

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

    return pd.DataFrame(rows).fillna(0.0)


def build_core_preference(x_norm: np.ndarray, keep_ratio: float = 0.4) -> np.ndarray:
    centroid = x_norm.mean(axis=0)
    dists = np.linalg.norm(x_norm - centroid, axis=1)
    keep_n = max(3, int(len(dists) * keep_ratio))
    keep_idx = np.argsort(dists)[:keep_n]
    return x_norm[keep_idx].mean(axis=0)


def main() -> None:
    df = load_person_feature_df(OUTPUT_DIR)
    if df.empty:
        print("[WARN] No person_profile.json found in output folder.")
        return

    feature_cols = [c for c in df.columns if c != "person"]
    x = df[feature_cols].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    x_norm = scaler.fit_transform(x)

    core_pref = build_core_preference(x_norm, keep_ratio=0.4)
    similarity = cosine_similarity(x_norm, core_pref.reshape(1, -1)).flatten()

    # Correlation analysis
    corr_rows = []
    for i, feat in enumerate(feature_cols):
        feat_values = x_norm[:, i]
        if np.std(feat_values) < 1e-9:
            corr = 0.0
        else:
            corr = float(np.corrcoef(feat_values, similarity)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        corr_rows.append({"feature": feat, "correlation": corr, "abs_correlation": abs(corr)})

    corr_df = pd.DataFrame(corr_rows).sort_values("abs_correlation", ascending=False)

    # RandomForest feature importance
    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    model.fit(x_norm, similarity)

    imp_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    # Save outputs
    corr_path = OUTPUT_DIR / "feature_correlation.csv"
    imp_path = OUTPUT_DIR / "feature_importance.csv"
    sim_path = OUTPUT_DIR / "core_similarity_scores.csv"
    summary_path = OUTPUT_DIR / "attraction_signal_summary.json"

    corr_df.to_csv(corr_path, index=False, encoding="utf-8")
    imp_df.to_csv(imp_path, index=False, encoding="utf-8")
    pd.DataFrame({"person": df["person"], "similarity_to_core": similarity}).sort_values(
        "similarity_to_core", ascending=False
    ).to_csv(sim_path, index=False, encoding="utf-8")

    summary = {
        "top_correlation_features": corr_df.head(12).to_dict(orient="records"),
        "top_importance_features": imp_df.head(12).to_dict(orient="records"),
        "notes": "Correlation shows linear relation direction; RF importance shows nonlinear predictive contribution.",
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved: {corr_path}")
    print(f"[DONE] Saved: {imp_path}")
    print(f"[DONE] Saved: {sim_path}")
    print(f"[DONE] Saved: {summary_path}")


if __name__ == "__main__":
    main()
