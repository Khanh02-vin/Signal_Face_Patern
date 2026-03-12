Signal Face Pattern
Overview

Signal Face Pattern is a computer vision pipeline designed to analyze visual attraction signals from videos or images.
The system extracts facial structure, behavioral signals, and emotional patterns, then builds a preference model to discover common patterns among people that the user finds attractive.

The pipeline processes raw media files, computes facial features, aggregates statistics, and produces ranking and clustering reports.

Main capabilities

Extract facial structure features

Analyze facial behavior and expression signals

Detect emotional patterns

Build preference vectors

Compute similarity rankings

Generate clustering insights

Produce final attraction reports

Input data is stored in raw/ and results are written to output/.

Project Structure
Signal-Face-Pattern/
│
├── attraction_pipeline/
│   ├── config.py
│   ├── dataset_io.py
│   ├── video_features.py
│   ├── analysis.py
│   └── pipeline.py
│
├── raw/
│   └── (input videos or images)
│
├── output/
│   └── (generated reports and rankings)
│
├── run_attraction_pipeline.py
├── build_ensemble_ranking.py
├── dashboard.py
├── requirements.txt
└── README.md
Folder description

attraction_pipeline/
Core pipeline modules.

File	Description
config.py	Pipeline configuration
dataset_io.py	Dataset loading and scanning
video_features.py	Feature extraction from videos
analysis.py	Similarity, clustering, and preference modeling
pipeline.py	Main pipeline orchestration

raw/
Input dataset containing videos or images.

Example:

raw/
   person_001/
       video1.mp4
       video2.mp4
   person_002/
       video1.mp4

output/
Generated results such as:

output/
   preference_strategy_similarity_scores.csv
   ensemble_taste_ranking.csv
   ensemble_taste_top50.csv
Installation
1. Create virtual environment
python -m venv venv
2. Activate environment

Windows

venv\Scripts\activate

macOS / Linux

source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
Running the Pipeline

Edit dataset paths if necessary in:

run_attraction_pipeline.py

Then run:

python run_attraction_pipeline.py

The pipeline will:

Scan dataset

Process videos

Extract facial features

Compute statistics

Generate similarity scores

Save results to output/

Build Ensemble Ranking

After generating similarity scores, run:

python build_ensemble_ranking.py

This script combines multiple scoring strategies into a final ranking.

Output files:

output/ensemble_taste_ranking.csv
output/ensemble_taste_top50.csv
Configuration

Important parameters are located in:

attraction_pipeline/config.py

Key settings include:

dataset paths

frame sampling rate

clustering parameters

similarity strategies

multiprocessing options

Example:

use_multiprocessing = True
max_workers = 4
frame_skip = 5
Performance Notes

For large datasets:

Enable multiprocessing

Adjust max_workers based on CPU cores

Increase frame_skip to reduce computation

Example recommendation:

Dataset size: 1000+ videos
Workers: 4–8
Frame skip: 5–10
Output Examples

Example ranking output:

person_id | similarity_score
----------|------------------
person_042 | 0.93
person_017 | 0.91
person_109 | 0.88

This represents how closely each person matches the learned attraction preference pattern.

Optional Dashboard

If available, run:

python dashboard.py

to visualize:

similarity rankings

clustering results

preference patterns
