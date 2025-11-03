# Music Genre Detector

A small Flask web app and model ensemble for predicting the genre of a music/audio file using audio feature extraction (via librosa) and scikit-learn classifiers.

## Dataset
Uses the [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)


## Features

- Upload an audio file (wav, mp3, flac, m4a, ogg) through the web UI and get a predicted genre.
- Programmatic API endpoint for predictions (`/api/predict`).
- Uses librosa to extract spectral, temporal, MFCC, chroma, contrast and tempo features.
- Ensemble of scikit-learn classifiers (models saved to `models/`).

## Requirements

- Python 3.8+ recommended
- Key Python packages (not exhaustive):
  - Flask
  - librosa
  - numpy
  - pandas
  - scikit-learn
  - joblib
  - soundfile (required by librosa)


## Install (local, using bash)

Create and activate a virtual environment, then install dependencies from the provided `requirements.txt`:

```bash
python -m venv venv
source venv/Scripts/activate

pip install --upgrade pip
pip install -r requirements.txt
```

If you prefer to install packages individually during development you can still use `pip install flask librosa numpy pandas scikit-learn joblib soundfile`, but installing from `requirements.txt` ensures compatible versions are used.


## Run the web app

Start the Flask app (dev mode):

```bash
# from repository root
python app.py
```


## API usage

POST an audio file to the `/api/predict` endpoint. Example (curl):

```bash
curl -X POST -F "file=@/path/to/audio.wav" -F "model=ensemble" http://localhost:5000/api/predict
```

Successful response JSON example:

```json
{
  "prediction": "rock",
  "confidence_scores": {"rock": 0.6, "pop": 0.2, "metal": 0.1},
  "model_used": "ensemble"
}
```

The web UI also allows selecting which model to use (if available): `rf`, `gb`, `svm`, `lr`, or `ensemble`.


