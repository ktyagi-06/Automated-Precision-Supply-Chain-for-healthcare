# Deployment guide

## 1. Deploy backend + model on Render
1. Push this repository to GitHub.
2. In Render, create a **Web Service** from the repository. Leave Root Directory blank (repository root).
3. Build command: `pip install -r backend/requirements.txt`
4. Start command: `uvicorn main:app --app-dir backend --host 0.0.0.0 --port $PORT`
5. Deploy. Test `https://YOUR-BACKEND.onrender.com/api/health` and `/docs`.

The backend reads the root-level `data` and `models` folders; keep them in the repository. The saved model and encoder are included. To retrain, run `pip install -r ml/requirements.txt` then `python ml/train_model.py` from the repository root.

## 2. Deploy frontend on Vercel
1. Import the same GitHub repository into Vercel. Set Root Directory to `frontend`.
2. Framework preset: Vite. Build command: `npm run build`. Output directory: `dist`.
3. Add environment variable `VITE_API_BASE` = `https://YOUR-BACKEND.onrender.com` (no trailing slash).
4. Redeploy after adding the environment variable.

## 3. Local run
Backend (from repository root): `cd backend`, install `pip install -r requirements.txt`, then `uvicorn main:app --reload --port 8000`.
Frontend (new terminal): `cd frontend`, `npm install`, then `npm run dev`. Vite proxies `/api` to localhost:8000.

## ML data note
The bundled sales and inventory data are synthetic demonstration data. Model metrics describe validation on that dataset and should not be presented as real hospital procurement performance. Replace the demo CSVs with appropriately governed real data before operational use.
