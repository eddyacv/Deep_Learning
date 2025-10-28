# ⚽ Football Event Recognition Project

This project aims to train a model to detect and classify football match events
(goals, fouls, substitutions, etc.) using the SoccerNet dataset.

## Structure
- `data/` → contains videos and annotations
- `scripts/` → preprocessing and training scripts
- `requirements.txt` → dependencies list

## Setup
```bash
pip install -r requirements.txt
python scripts/download_soccer.py
python scripts/preprocess_soccer.py
```
