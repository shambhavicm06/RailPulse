# 🚀 Deploy RailPulse to Render (free tier)

Get a **permanent public HTTPS link** (e.g. `https://railpulse.onrender.com`) that
stays up 24/7. The free tier "sleeps" after ~15 minutes of inactivity and wakes
up automatically on the next visit (first visit after sleep takes ~30–60 s).

---

## What you need (one-time, ~5 minutes)

1. A **GitHub** account + a new (public or private) repository.
2. A **Render** account — sign up at https://render.com with your GitHub login.

---

## Step 1 — Push the project to GitHub

On your Windows machine, open a terminal inside the project folder
(`railway-cascade-predictor`) and run:

```bat
git init
git add .
git commit -m "RailPulse railway cascade predictor"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<REPO>.git
git push -u origin main
```

What gets pushed (and used by Render):

| Path | Purpose |
|------|---------|
| `Dockerfile` | Builds the container image |
| `render.yaml` | Render Blueprint (one-click deploy) |
| `deploy/requirements.txt` | Lean Python deps (no Gradio; CatBoost included) |
| `src/` | FastAPI backend + HTML dashboard + ML pipeline |
| `lib/` | Leaflet.js + map icons (served locally) |
| `data/` | Synthetic dataset + sample input CSV |
| `results/` | Metrics, ablations, charts shown in Analytics |
| `models/bundle.joblib` | Trained 10-model bundle (instant boot, ~30 MB) |

> `exports/`, `__pycache__/` and zips are excluded by `.gitignore`.

---

## Step 2 (recommended) — Deploy via Blueprint

1. Open **Render Dashboard** → **New +** → **Blueprint**.
2. Connect your GitHub account and select the repo.
3. Render reads `render.yaml` and shows one web service named **railpulse**.
4. Click **Apply**. It builds the Docker image (a few minutes) and deploys.

---

## Step 2 (alternative) — Manual Web Service

1. **New +** → **Web Service** → select your repo.
2. Render auto-detects the `Dockerfile`. Set:
   - Name: `railpulse`
   - Instance type: **Free**
3. Under **Advanced**:
   - Health Check Path: `/health`
   - Environment variables: `SWR_USER=admin`, `SWR_PASS=swr2026`
4. Click **Create Web Service**.

---

## Step 3 — Verify

- Wait for the build to finish (watch the deploy logs).
- Open `https://<service-name>.onrender.com`.
- Sign in with **`admin` / `swr2026`** (or the `SWR_USER` / `SWR_PASS` you set).
- Open the **Network Graph** tab — the railway network and real map tiles load.
  Tiles are served same-origin by the app (`/tiles/...`), so the visitor's
  browser does **not** need direct access to OpenStreetMap/Esri.

---

## Customising

- **Change login**: edit `SWR_USER` / `SWR_PASS` in the Blueprint (`render.yaml`)
  or in Render → your service → **Environment**.
- **Enable the 🤖 Dispatcher Copilot's LLM layer** (optional): in Render →
  your service → **Environment**, add:
  - `COPILOT_LLM_KEY` = your OpenAI-compatible API key
  - `COPILOT_LLM_BASE` = e.g. `https://api.openai.com/v1` (or Groq/Together/Ollama)
  - `COPILOT_LLM_MODEL` = e.g. `gpt-4o-mini`
  Without these, the Copilot still works in offline (deterministic) mode.
- **Smaller repo / image**: remove `models/` from the Dockerfile and
  `COPY models ./models` line — the app auto-retrains on first boot (~2–3 min
  on the free tier). This is safe: the training pipeline is fully self-healing.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Build fails on `pip install` | Check deploy logs; usually a transient network issue — click **Manual Deploy → Deploy latest commit** |
| "Your free instance will spin down" | Normal — visit the URL to wake it up |
| Page loads but map is blank | Browser has no internet for OSM tiles; network/markers still render on the fallback background |
| Login says "Backend offline" | Wait for the service to finish starting (bundle load takes ~10–20 s) and refresh |
