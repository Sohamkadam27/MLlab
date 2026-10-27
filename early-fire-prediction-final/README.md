# Early Forest Fire Prediction: Fire Risk Map + Prediction Panel

Live wildfire risk for the **Santa Clara County foothills (California)**, combining
1. **Real satellite fire detections** (NASA FIRMS VIIRS), live with a free key, or a replay of real past days without one.
2. **Live weather and soil conditions** (Open-Meteo), scored by an **XGBoost** model trained on 2020–2023 data.

## Project layout
```
data_utils.py  train_xgboost.py  train_other_models.py  calibrate_thresholds.py   <- ML scripts (project root)
Trihourly_Weather_Dataset.csv                                                     <- training data (you add)
backend/   FastAPI: main.py, routes/, services/, model/ (generated), data/ (you add the FIRMS csv)
frontend/  React + Vite: Fire Risk Map and Prediction Panel
```

## 1. One-time setup (Windows PowerShell)
```powershell
# backend
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cd ..
pip install -r requirements-training.txt

# put these two files in place
#   Trihourly_Weather_Dataset.csv            -> project root
#   Wildfire_Data_2020-2023.csv              -> backend/data/

# train + calibrate (about 10 seconds). ALWAYS run both, in this order
python train_xgboost.py
python calibrate_thresholds.py

# frontend
cd frontend
npm install
```

## 2. Run
```powershell
# terminal 1
cd backend ; venv\Scripts\activate ; uvicorn main:app --reload
# terminal 2
cd frontend ; npm run dev          # http://localhost:5173
```
Check **http://127.0.0.1:8000/status**. It should say `"ok": true`. If anything is wrong it lists the problem.

## 3. Optional: live satellite data
Get a free key at https://firms.modaps.eosdis.nasa.gov/api/map_key/ , then copy `backend/.env.example` to
`backend/.env` and set `FIRMS_MAP_KEY=...`. Without a key the map **replays** real detections from
`Wildfire_Data_2020-2023.csv` (default day 2020-08-20, the peak of the CZU/SCU fires); the map shows a
REPLAY badge so it is never mistaken for live data. Any day: `/hotspots?date=2020-08-21`; list days: `/hotspots/dates`.

## API
| Endpoint | Purpose |
|---|---|
| `GET /status` | health check, model validation numbers, hotspot mode |
| `GET /zones` | live conditions + model risk + score drivers for each monitored zone |
| `GET /hotspots`, `/heatmap`, `/hotspots/dates` | satellite detections (live or replay) |
| `POST /predict` | legacy 3-input estimate. **Unreliable** (see below), kept for compatibility |

## How the model was evaluated (honest numbers)
Trained on 2020–2022, tested on the **held-out year 2023** (a random split would leak near-duplicate
3-hourly rows and inflate scores).

| | ROC AUC | PR AUC (random = 0.084) |
|---|---|---|
| XGBoost (final) | 0.987 | 0.963 |
| Gradient Boosting / SVM | 0.991 / 0.986 | 0.963 / 0.963 |
| Logistic Regression | 0.928 | 0.789 |

Run `python train_other_models.py` to reproduce the table. Risk levels use cutoffs learned from the 2023 holdout
(`calibrate_thresholds.py`): **High** is about 93% precise and catches about 95% of fire rows; **Low** (the normal state, 90% of days) contains 0.3% fire rows; **Medium** is a rare, narrow band (about 1% of days) where roughly 1 in 5 rows is a fire.

## Limitations (please read before presenting)
- **One training site.** Weather came from a single point (37.3 N, 121.5 W, see `final_dataset_centroid.ipynb` in the
  original repo) and a day is labelled "fire" if FIRMS saw a hotspot anywhere in 37.1–37.5 N, 121.8–121.2 W. The model estimates
  *regional fire activity today*, not independent per-site risk. Zones differ only through their own live readings.
- **The model is driven mostly by surface pressure and soil conditions** (see "What drives this score"). They act as a seasonal
  signal; air temperature/humidity/wind alone carry little information (3-input AUC about 0.53, hence `/predict` is unreliable).
- **Scores are strongly bimodal.** The model is either confident there is no fire activity (Low) or confident there is (High); Medium is rare. A mostly-green map on an ordinary day is the expected, correct result.
- Live soil layers differ from the ERA5 layers used for training and pressure is rescaled to the training site, so live scores are indicative.
- 2020 is heavily over-represented in training (a record fire season), so raw model scores are not literal probabilities.

## Troubleshooting
| Symptom | Fix |
|---|---|
| Red "Backend unreachable" banner | start uvicorn; check `VITE_API_URL` |
| "Risk zones hidden: ..." banner | `/zones` failed: open `http://127.0.0.1:8000/zones` and read the `error` field |
| `/status` says model not ready | run `train_xgboost.py` then `calibrate_thresholds.py` |
| Sklearn/xgboost pickle warning | retrain in the current venv |
| Hotspot count is 0 with a key set | genuinely no fires nearby; set `HOTSPOT_MODE=replay` to demo |

Deploying: set `VITE_API_URL` in `frontend/.env.production` to **your** backend before `npm run build`.
