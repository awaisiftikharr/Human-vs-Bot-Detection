# Human vs. Bot Detection

A Streamlit demonstration that classifies a short behavioural session using three numeric features: pointer movement distance, elapsed interaction time, and movement jitter.

## Features

- Interactive Streamlit interface
- Scikit-learn model loaded from `bot_detector.pkl`
- Input validation and consistent feature names
- Optional prediction confidence when the model supports probabilities
- Reproducible model-training script using the checked-in CSV dataset

## Tech stack

Python · pandas · scikit-learn · joblib · Streamlit · pytest

## Dataset

`behavior_data.csv` contains 2,000 rows and the columns:

- `Distance`
- `Time`
- `Jitter`
- `Is_Human`

The repository's original app convention maps `Is_Human = 1` to human and `Is_Human = 0` to bot. Confirm label provenance and collection methodology before using the data for anything beyond a demonstration.

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Retrain the model

```bash
python train_model.py
```

The script performs a stratified train/test split, reports held-out metrics, and saves a new `bot_detector.pkl` artifact. Review the metrics and data provenance before using the model.

## Run tests

```bash
pytest -q
```

## Limitations

This is a small-feature demonstration, not a production bot-detection system. Three aggregate signals can be spoofed and may unfairly classify atypical human behaviour. Production use would require consented and representative telemetry, careful privacy review, robust validation, calibration, adversarial testing, and human review.
