# Signal Face Pattern

## Tổng quan dự án
Signal Face Pattern là một dự án **AI/ML** tập trung vào việc trích xuất **facial signals** từ video, tổng hợp thành **person profile**, và phân tích **attraction pattern** dựa trên các đặc trưng hình học, hành vi và cảm xúc khuôn mặt. Pipeline hỗ trợ xử lý dữ liệu video theo thư mục người dùng, tính toán **feature statistics**, sau đó thực hiện **clustering** và tạo báo cáo phân tích.

## Mục tiêu chính
Giải quyết bài toán: **tự động hoá việc phân tích pattern hấp dẫn dựa trên tín hiệu khuôn mặt từ video**, bao gồm mức độ tương đồng, nhóm cụm và tóm tắt đặc trưng nổi bật.

## Công nghệ sử dụng
- **Python**
- **OpenCV** (video processing)
- **MediaPipe** (face mesh landmarks)
- **DeepFace** (emotion analysis)
- **NumPy / Pandas**
- **scikit-learn** (clustering, PCA)
- **Matplotlib** (visualization)

## Tính năng chính
- Trích xuất **facial geometry** và **behavior signals** từ video
- Tổng hợp **person profile** và đánh giá **quality score**
- Phân tích **preference vector** và **similarity scores**
- **Clustering** bằng KMeans/DBSCAN/auto_kmeans
- Xuất báo cáo JSON/CSV và biểu đồ (Radar, PCA)

## Cài đặt
### Yêu cầu
- Python 3.9+

### Cài dependencies

```bash
pip install -r requirements.txt
```

## Hướng dẫn sử dụng
### 1) Chuẩn bị dataset
Tổ chức thư mục `raw/` theo cấu trúc **person-level**:

```
raw/
├── person_01/
│   ├── video_01.mp4
│   ├── video_02.mp4
├── person_02/
│   ├── video_01.mp4
│   ├── video_02.mp4
```

### 2) Chạy pipeline
Chỉnh sửa đường dẫn trong `run_attraction_pipeline.py` nếu cần, sau đó chạy:

```bash
python run_attraction_pipeline.py
```

### 3) Kết quả đầu ra
Các file sẽ được ghi trong thư mục `output/`, bao gồm:
- `attraction_report.json`
- `similarity_scores.csv`
- `person_quality_scores.csv`
- `attraction_radar.png`
- `pca_person_vectors.png`

## Cấu trúc dự án

```
Signal_Face_Patern/
├── attraction_pipeline/
│   ├── __init__.py
│   ├── analysis.py
│   ├── config.py
│   ├── dataset_io.py
│   ├── pipeline.py
│   └── video_features.py
├── lib/
│   ├── bindings/
│   ├── tom-select/
│   └── vis-9.1.2/
├── output/
├── raw/
├── attraction_signal_analysis.py
├── build_ensemble_ranking.py
├── build_final_core_taste_report
├── dashboard.py
├── person_to_person_similarity.py
├── preference_benchmark.py
├── preference_methods.py
├── preference_strategies.py
├── run_attraction_pipeline.py
└── requirements.txt
```

## Ví dụ code
Ví dụ chạy pipeline đơn giản:

```python
from pathlib import Path
from attraction_pipeline.config import PipelineConfig
from attraction_pipeline.pipeline import run_pipeline

config = PipelineConfig(
    dataset_root=Path("raw"),
    output_root=Path("output"),
    frame_step=5,
    min_face_confidence=0.5,
    max_frames_per_video=None,
    cluster_k=3,
    cluster_method="auto_kmeans",
    dbscan_eps=1.2,
    dbscan_min_samples=3,
    use_multiprocessing=False,
)

run_pipeline(config)
```

## Dataset
Dự án sử dụng **video dataset** được sắp theo từng thư mục người dùng. Mỗi thư mục chứa nhiều video của cùng một người. Pipeline sẽ xử lý từng video, trích xuất **frame-level signals** và tổng hợp thành **person-level profile**.

## Model Architecture (Pipeline Architecture)
Pipeline chính gồm các bước:
1. **Video Feature Extraction**: dùng MediaPipe để lấy face landmarks, OpenCV để đọc video, DeepFace để suy luận emotion.
2. **Aggregation**: tổng hợp feature thống kê (mean, std, median, max) theo video và theo người.
3. **Analysis & Clustering**: chuẩn hoá, tạo preference vector, cosine similarity, clustering, PCA.
4. **Reporting**: xuất JSON/CSV và visualization.

## Kết quả/Outputs
- **Attraction report**: tổng hợp preference vector, similarity score, cluster assignment.
- **CSV output**: chất lượng dữ liệu và điểm tương đồng.
- **Visualization**: radar chart và PCA scatter plot.

## Future Improvements
- Bổ sung **temporal modeling** (LSTM/Transformer) cho sequence features
- Tối ưu **speed** với batch processing và GPU acceleration
- Thêm module **face quality filtering** nâng cao
- Cải thiện dashboard và reporting trực quan
