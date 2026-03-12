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

        with profile_path.open("r", encoding="utf-8") as f:
            profile = json.load(f)

        row = {"person": person_dir.name}
        for k, v in profile.items():
            if isinstance(v, (int, float)) and k not in NON_FEATURE_COLUMNS:
                row[k] = float(v)
        rows.append(row)

        quality_rows.append(
            {
                "person": person_dir.name,
                "quality_score": float(profile.get("quality_score", 0.5)),
                "valid_video_ratio": float(profile.get("valid_video_ratio", 0.5)),
                "avg_processed_frames": float(profile.get("avg_processed_frames", 0.0)),
            }
        )

    return pd.DataFrame(rows).fillna(0.0), pd.DataFrame(quality_rows).fillna(0.0)


def build_methods(x_norm: np.ndarray, quality_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    # mean_all
    mean_vec = x_norm.mean(axis=0)

    # weighted_mean
    q = quality_df["quality_score"].to_numpy(dtype=np.float32)
    v = quality_df["valid_video_ratio"].to_numpy(dtype=np.float32)
    f = quality_df["avg_processed_frames"].to_numpy(dtype=np.float32)
    if np.max(f) > 0:
        f = f / np.max(f)
    w = 0.6 * q + 0.3 * v + 0.1 * f
    w = np.clip(w, 1e-6, None)
    w = w / np.sum(w)
    weighted_vec = np.average(x_norm, axis=0, weights=w)

    # core_percentile_40
    c = x_norm.mean(axis=0)
    d = np.linalg.norm(x_norm - c, axis=1)
    keep_n = max(3, int(len(d) * 0.4))
    keep_idx = np.argsort(d)[:keep_n]
    core_vec = x_norm[keep_idx].mean(axis=0)

    # largest_cluster_center
    n_clusters = max(2, min(3, len(x_norm)))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = km.fit_predict(x_norm)
    dom = int(pd.Series(labels).value_counts().index[0])
    dom_idx = np.where(labels == dom)[0]
    cluster_vec = x_norm[dom_idx].mean(axis=0)

    return {
        "mean_all": mean_vec,
        "weighted_mean": weighted_vec,
        "core_percentile_40": core_vec,
        "largest_cluster_center": cluster_vec,
    }


def cohesion_score(x_norm: np.ndarray, pref_vec: np.ndarray) -> float:
    sims = cosine_similarity(x_norm, pref_vec.reshape(1, -1)).flatten()
    top_n = max(5, int(len(sims) * 0.2))
    idx = np.argsort(sims)[::-1][:top_n]
    group = x_norm[idx]
    if len(group) < 2:
        return 0.0
    pair_sim = cosine_similarity(group)
    upper = pair_sim[np.triu_indices(len(group), k=1)]
    return float(np.mean(upper)) if upper.size > 0 else 0.0


def topk_consistency(x_norm: np.ndarray, pref_vec: np.ndarray, k: int = 20, n_boot: int = 25) -> float:
    base = cosine_similarity(x_norm, pref_vec.reshape(1, -1)).flatten()
    base_top = set(np.argsort(base)[::-1][:k].tolist())
    if len(base_top) == 0:
        return 0.0

    rng = np.random.default_rng(42)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(len(x_norm), size=len(x_norm), replace=True)
        xb = x_norm[idx]
        sims = cosine_similarity(xb, pref_vec.reshape(1, -1)).flatten()
        boot_top_idx = np.argsort(sims)[::-1][:k]
        mapped = set(idx[boot_top_idx].tolist())
        inter = len(base_top & mapped)
        union = len(base_top | mapped)
        scores.append(inter / union if union > 0 else 0.0)
    return float(np.mean(scores))


def bootstrap_stability(x_norm: np.ndarray, method_name: str, quality_df: pd.DataFrame, n_boot: int = 30) -> float:
    rng = np.random.default_rng(123)
    vectors = []

    for _ in range(n_boot):
        idx = rng.choice(len(x_norm), size=len(x_norm), replace=True)
        xb = x_norm[idx]

        if method_name == "mean_all":
            v = xb.mean(axis=0)

        elif method_name == "weighted_mean":
            qdf = quality_df.iloc[idx].reset_index(drop=True)
            q = qdf["quality_score"].to_numpy(dtype=np.float32)
            vr = qdf["valid_video_ratio"].to_numpy(dtype=np.float32)
            fr = qdf["avg_processed_frames"].to_numpy(dtype=np.float32)
            if np.max(fr) > 0:
                fr = fr / np.max(fr)
            w = 0.6 * q + 0.3 * vr + 0.1 * fr
            w = np.clip(w, 1e-6, None)
            w = w / np.sum(w)
            v = np.average(xb, axis=0, weights=w)

        elif method_name == "core_percentile_40":
            c = xb.mean(axis=0)
            d = np.linalg.norm(xb - c, axis=1)
            keep_n = max(3, int(len(d) * 0.4))
            keep_idx = np.argsort(d)[:keep_n]
            v = xb[keep_idx].mean(axis=0)

        else:  # largest_cluster_center
            n_clusters = max(2, min(3, len(xb)))
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            labels = km.fit_predict(xb)
            dom = int(pd.Series(labels).value_counts().index[0])
            dom_idx = np.where(labels == dom)[0]
            v = xb[dom_idx].mean(axis=0)

        vectors.append(v)

    vectors = np.array(vectors)
    center = vectors.mean(axis=0)
    dists = np.linalg.norm(vectors - center, axis=1)
    # smaller dispersion -> higher stability
    return float(1.0 / (1.0 + np.mean(dists)))


def minmax_scale(values: Dict[str, float]) -> Dict[str, float]:
    arr = np.array(list(values.values()), dtype=np.float32)
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    if vmax <= vmin + 1e-12:
        return {k: 0.0 for k in values}
    return {k: float((v - vmin) / (vmax - vmin)) for k, v in values.items()}


def main() -> None:
    feature_df, quality_df = load_person_profiles(OUTPUT_DIR)
    if feature_df.empty:
        print("[WARN] No person profiles in output.")
        return

    feature_cols = [c for c in feature_df.columns if c != "person"]
    x = feature_df[feature_cols].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    x_norm = scaler.fit_transform(x)

    methods = build_methods(x_norm, quality_df)

    cohesion_raw = {}
    consistency_raw = {}
    stability_raw = {}

    for name, vec in methods.items():
        cohesion_raw[name] = cohesion_score(x_norm, vec)
        consistency_raw[name] = topk_consistency(x_norm, vec, k=min(20, len(x_norm)))
        stability_raw[name] = bootstrap_stability(x_norm, name, quality_df, n_boot=25)

    cohesion_n = minmax_scale(cohesion_raw)
    consistency_n = minmax_scale(consistency_raw)
    stability_n = minmax_scale(stability_raw)

    rows = []
    final_scores = {}
    for name in methods.keys():
        final_score = 0.4 * cohesion_n[name] + 0.35 * stability_n[name] + 0.25 * consistency_n[name]
        final_scores[name] = final_score
        rows.append(
            {
                "method": name,
                "cohesion_raw": cohesion_raw[name],
                "stability_raw": stability_raw[name],
                "consistency_raw": consistency_raw[name],
                "cohesion_norm": cohesion_n[name],
                "stability_norm": stability_n[name],
                "consistency_norm": consistency_n[name],
                "composite_score": final_score,
            }
        )

    score_df = pd.DataFrame(rows).sort_values("composite_score", ascending=False).reset_index(drop=True)
    best_method = str(score_df.iloc[0]["method"])

    report = {
        "best_method": best_method,
        "scoring_formula": "0.4*cohesion + 0.35*stability + 0.25*consistency (normalized)",
        "methods": score_df.to_dict(orient="records"),
    }

    score_df.to_csv(OUTPUT_DIR / "preference_method_scores.csv", index=False, encoding="utf-8")
    with (OUTPUT_DIR / "preference_benchmark_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[DONE] output/preference_method_scores.csv")
    print("[DONE] output/preference_benchmark_report.json")
    print(f"[DONE] Best method: {best_method}")


if __name__ == "__main__":
    main()