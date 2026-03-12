from __future__ import annotations

import pandas as pd
from pathlib import Path

OUT = Path("output")
sim_path = OUT / "preference_strategy_similarity_scores.csv"
df = pd.read_csv(sim_path)

# normalize each score column to [0,1]
for col in ["core_percentile_40", "largest_cluster_center"]:
    cmin, cmax = df[col].min(), df[col].max()
    if cmax > cmin:
        df[f"{col}_norm"] = (df[col] - cmin) / (cmax - cmin)
    else:
        df[f"{col}_norm"] = 0.0

# ensemble weight: robust core 0.65 + dominant cluster 0.35
df["ensemble_score"] = 0.65 * df["core_percentile_40_norm"] + 0.35 * df["largest_cluster_center_norm"]

ranked = df.sort_values("ensemble_score", ascending=False).reset_index(drop=True)
ranked.to_csv(OUT / "ensemble_taste_ranking.csv", index=False, encoding="utf-8")
ranked.head(50).to_csv(OUT / "ensemble_taste_top50.csv", index=False, encoding="utf-8")

print("[DONE] Saved: output/ensemble_taste_ranking.csv")
print("[DONE] Saved: output/ensemble_taste_top50.csv")