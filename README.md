# Signal Face Pattern

## Overview
Trích xuất **facial signals** từ video, tổng hợp thành **person profile**, và phân tích **attraction pattern** dựa trên facial geometry, hành vi và cảm xúc khuôn mặt.

## Results

| Metric | Value |
|--------|-------|
| Dataset | 164 celebrities, 1,268 videos |
| Features | 39 signal features per person |
| Clusters | 2 clusters (auto KMeans) |
| Top similarity | Bạch Băng (0.557), Roh Jeong Un (0.535), Fujiyoshi Karin (0.522) |

## Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | `cluster_0` | High facial symmetry, strong expression dynamics |
| 1 | `cluster_1` | Lower symmetry, more neutral expressions |

## Dataset

> **Source:** [Khanh510/Signal_Face](https://huggingface.co/datasets/Khanh510/Signal_Face)
>
> **Stats:** 164 celebrities | 1,268 videos | 39 features

## Setup

```bash
pip install -r requirements.txt
```

Organize `raw/` by person:
```
raw/
├── person_01/
│   ├── video_01.mp4
├── person_02/
│   ├── video_01.mp4
```

## Usage

### Full pipeline (requires raw video)
```bash
python run_attraction_pipeline.py
```

### Analysis only (pre-processed CSVs — no video needed)
```bash
python run_analysis_from_csv.py
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Video processing | OpenCV |
| Face mesh | MediaPipe Face Mesh |
| Emotion analysis | DeepFace |
| Clustering | scikit-learn (KMeans, DBSCAN, PCA) |
| Visualization | Matplotlib, HTML dashboard |

## Output Files

| File | Description |
|------|-------------|
| `attraction_report.json` | Full analysis with preference vectors |
| `similarity_scores.csv` | Cosine similarity per person to centroid |
| `person_quality_scores.csv` | Data quality scores |
| `attraction_radar.png` | Radar chart (face/expression/emotion) |
| `pca_person_vectors.png` | PCA 2D scatter by cluster |

## Architecture

```
Video Input
    ↓
MediaPipe Face Mesh → 468 facial landmarks
DeepFace → Emotion distribution (Happy, Neutral, Surprise...)
OpenCV → Eye movement, head pose
    ↓
Per-video feature statistics (mean, std, median, max)
    ↓
Person-level aggregation (39 features)
    ↓
KMeans / DBSCAN / PCA
    ↓
JSON report + CSV + Radar/Scatter visualization
```

## Key Results

**Top attraction patterns:**
- `Bạch Băng` — similarity=0.557, cluster=0, 7 videos
- `Roh Jeong Un` — similarity=0.535, cluster=1, 7 videos
- `Fujiyoshi Karin` — similarity=0.522, cluster=0, 7 videos
- `Park Jihyun` — similarity=0.489, cluster=0, 8 videos

**Signal categories:** facial symmetry, eye metrics, smile intensity, emotion distribution, head pose variation
