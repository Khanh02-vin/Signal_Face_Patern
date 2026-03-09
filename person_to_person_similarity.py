from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


NON_FEATURE_COLUMNS = {
    "person_name",
    "video_count",
    "valid_video_count",
    "valid_video_ratio",
    "total_processed_frames",
    "avg_processed_frames",
    "quality_score",
}


def load_person_profiles(output_root: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for person_dir in sorted(output_root.iterdir()):
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


def build_similarity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c != "person"]
    X = df[feature_cols].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)

    sim = cosine_similarity(Xn)
    sim_df = pd.DataFrame(sim, index=df["person"], columns=df["person"])
    return sim_df


def build_top_pairs(sim_df: pd.DataFrame, top_k: int = 300) -> pd.DataFrame:
    persons = sim_df.index.tolist()
    records: List[Dict] = []

    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            p1 = persons[i]
            p2 = persons[j]
            records.append(
                {
                    "person_a": p1,
                    "person_b": p2,
                    "similarity": float(sim_df.loc[p1, p2]),
                }
            )

    pairs_df = pd.DataFrame(records).sort_values(by="similarity", ascending=False)
    return pairs_df.head(top_k)


def main() -> None:
    output_root = Path(r"D:\AI-ML\New folder\Signal_Face_Patern\output")

    df = load_person_profiles(output_root)
    if df.empty:
        print("[WARN] No person_profile.json files found.")
        return

    sim_df = build_similarity_matrix(df)
    top_pairs_df = build_top_pairs(sim_df, top_k=300)

    matrix_csv = output_root / "person_to_person_similarity_matrix.csv"
    top_pairs_csv = output_root / "person_to_person_top_pairs.csv"
    json_path = output_root / "person_to_person_similarity_summary.json"

    sim_df.to_csv(matrix_csv, encoding="utf-8")
    top_pairs_df.to_csv(top_pairs_csv, index=False, encoding="utf-8")

    summary = {
        "person_count": int(len(sim_df.index)),
        "matrix_csv": str(matrix_csv),
        "top_pairs_csv": str(top_pairs_csv),
        "most_similar_pairs": top_pairs_df.head(20).to_dict(orient="records"),
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved matrix: {matrix_csv}")
    print(f"[DONE] Saved top pairs: {top_pairs_csv}")
    print(f"[DONE] Saved summary: {json_path}")


if __name__ == "__main__":
    main()
