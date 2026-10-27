# Deployment guide

## 1. Backend on Render

1. Push this repository to GitHub.
2. In Render, create a **Blueprint** from the repo (it reads `render.yaml`), or a **Web Service**
   with build command `pip install -r backend/requirements.txt` and start command
   `uvicorn main:app --app-dir backend --host 0.0.0.0 --port $PORT`. Leave Root Directory blank.
3. Set `API_KEY` to a secret if you will use `/api/ingest/*`.
4. Check `https://YOUR-BACKEND.onrender.com/api/health` and `/docs`.

Things to know:

- **Live state is stored in a local SQLite file.** On Render's free tier the disk is
  ephemeral, so after a restart or redeploy the simulation starts fresh from the bundled
  history. To keep state, attach a persistent disk and set `LIVE_DB_PATH` to a path on it.
- **The free tier sleeps when idle**, which pauses the live clock. A paid instance stays awake.
- **Retraining is CPU-heavy** (about 12 s on 2 cores; slower on small instances). Keep
  `RETRAIN_FAST=true`, or set `RETRAIN_EVERY_DAYS=0` and retrain manually.
- If Render installs different scikit-learn or XGBoost versions than the committed model was
  built with, the server detects this at startup and trains a new model automatically.
- The live stream is Server-Sent Events over normal HTTP, so no special proxy setup is needed.

## 2. Frontend on Vercel

1. Import the same repository into Vercel and set Root Directory to `frontend`.
2. Framework preset: Vite. Build command: `npm run build`. Output directory: `dist`.
3. Add the environment variable `VITE_API_BASE` = `https://YOUR-BACKEND.onrender.com`
   (no trailing slash), then redeploy.
4. Optionally restrict the backend with `CORS_ORIGINS=https://YOUR-APP.vercel.app`.

## 3. Local run

```bash
pip install -r requirements-dev.txt
cd backend && uvicorn main:app --reload --port 8000
cd frontend && npm install && npm run dev
```

Retrain from scratch: `python ml/train_model.py`. Regenerate the demo data first with
`python ml/generate_data.py` if you changed the product profiles.

## Data note

The bundled sales and inventory data are synthetic demonstration data. Model metrics describe
validation on that dataset and should not be presented as real hospital procurement performance.
Replace the demo CSVs with appropriately governed real data before operational use.
