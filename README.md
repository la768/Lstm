# LSTM / Ensemble Prediction Models

An OCR-driven prediction system that reads live game results from the screen and uses an ensemble of machine-learning classifiers to forecast the next outcome.

## Overview

The project captures specific screen regions with `pyautogui`, performs OCR with **Tesseract**, **RapidOCR** and **EasyOCR**, and feeds the extracted history into an ensemble of models to predict:

- **Big / Small** — whether the result classifies as "big" or "small"
- **Number** — the exact numeric result
- **Color** — the result colour class

Prediction combinations are boosted/validated with time-series and pattern-based techniques (HMM, Markov, n-gram and regime/trend detectors) and a final consensus vote decides the output.

## Files

| File | Description |
| ---- | ----------- |
| `old.py` | Main OCR + ensemble training/prediction loop (screenshot capture, OCR, training, voting). |
| `patch_log_row.py` | Extended variant with a multi-agent decision log (entropy / pattern / bootstrap / ZK-matrix agents) and patched row logging. |
| `Ai_folder/ocr_result2.csv` | Captured result history (period, number, big/small, color). |
| `Ai_folder/dpae_explain.csv` | Detailed per-round decision/explanation log from the multi-agent pipeline. |

## Requirements

The scripts use a wide set of ML / OCR libraries. Key dependencies:

- `pyautogui`, `pytesseract`, `Pillow`, `opencv-python`, `numpy`, `pandas`
- `easyocr`, `rapidocr-onnxruntime`
- `scikit-learn`, `catboost`, `lightgbm`, `xgboost` (optional), `hmmlearn`
- `joblib`

> Note: Tesseract must be installed and the path set to `C:\Program Files\Tesseract-OCR\tesseract.exe` in the scripts.

## Models

Trained model artifacts (`.pkl`) are **git-ignored** and regenerated at runtime by `train_models()`:

- `models.pkl` — Big/Small + Number ensemble models
- `number_models.pkl` — number-specific ensemble
- `color_models.pkl` — color ensemble
- `scaler.pkl` — feature standardizer

## Usage

Run either script from the project root:

```bash
python old.py
# or
python patch_log_row.py
```

## License

This project is for personal/educational use.