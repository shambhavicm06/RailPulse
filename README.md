# 🚄 RailPulse — Railway Cascade Prediction System

**Graph-Derived Feature Boosting for Cascading Train Delay Prediction on Indian Railways**

RailPulse is an end-to-end, ML-powered **dispatcher workstation** for Indian Railways
(centred on the South Western Railway zone, extended to a 84-station national network).
It predicts how a delay at one station *cascades* through the rail network, ranks the
junctions most likely to choke, recommends mitigations, and lets you talk to the network
in plain English through a built-in **AI Dispatcher Copilot** — all on a real,
Google-Maps-style interactive map.

> **Headline result:** graph-derived features cut delay-prediction error (**MAE**) by
> ~46–53% versus isolated features alone, and a "radius-of-influence" ablation proves the
> delay ripple **decays with distance** — *no train is an island.*

---

## Table of contents

1. [What is RailPulse?](#1-what-is-railpulse)
2. [What you need (prerequisites)](#2-what-you-need-prerequisites)
3. [Getting started — from the unzipped folder](#3-getting-started--from-the-unzipped-folder)
4. [Signing in](#4-signing-in)
5. [Feature guide — every screen explained](#5-feature-guide--every-screen-explained)
6. [The interactive map (how to use it)](#6-the-interactive-map-how-to-use-it)
7. [How the ML works (the science)](#7-how-the-ml-works-the-science)
8. [Reproducing the full pipeline](#8-reproducing-the-full-pipeline)
9. [API reference](#9-api-reference)
10. [The Dispatcher Copilot (AI agent)](#10-the-dispatcher-copilot-ai-agent)
11. [Deploying to the cloud (Render)](#11-deploying-to-the-cloud-render)
12. [Running with Docker](#12-running-with-docker)
13. [Troubleshooting](#13-troubleshooting)
14. [Project structure](#14-project-structure)

---

## 1. What is RailPulse?

A saturated railway network is a *cascading system*: a 45-minute delay at a busy
junction monopolises track clearance and forces secondary trains to hold, which delays
more trains, and so on. Conventional delay models predict each train **in isolation** and
miss this ripple.

RailPulse treats the railway as a **graph** (stations = nodes, tracks = weighted edges),
flattens NetworkX graph analytics (degree, eigenvector centrality, PageRank, betweenness)
into tabular features, and feeds them to **10 classical ML models** — no GNNs required.
The winning **Network-Aware Voting Regressor** powers the live dashboard.

**Everything in one package:**

- 🔐 **Login-gated web app** (username/password)
- ☀️ Professional **light theme** by default (🌙 dark mode available) — modern cards, clear typography and friendly labels throughout
- 🕸️ **Real interactive map** (Leaflet + OpenStreetMap / Esri tiles) — zoom, pan, fullscreen
- 🎛️ **Delay prediction** with a colour-coded **Grid Cascade Alert**
- 🗂️ **Batch prediction** from CSV upload
- 🧠 **Dispatcher Copilot** — an AI agent you can chat with (offline, or LLM-powered), with 🎙 voice input
- 📊 **Model bake-off, ablations & analytics**
- 🚀 **Deployable to Render** in one click

---

## 2. What you need (prerequisites)

| Requirement | Details |
|-------------|---------|
| **Python** | 3.10 – 3.13 (3.11 recommended). Windows: tick **"Add Python to PATH"** during install. |
| **Disk** | ~250 MB free (mostly Python packages) |
| **Internet** | Needed once to install packages, and to load map tiles in the browser (the map uses real OpenStreetMap / Esri tiles). The app itself runs fully offline after setup. |
| **OS** | Windows 10/11, macOS, or Linux |

The trained model (`models/bundle.joblib`) is **included**, so the app runs immediately —
no training required on first launch.

---

## 3. Getting started — from the unzipped folder

### 3.1 Unzip

Extract `RailPulse.zip` anywhere (e.g. `C:\RailPulse`). You'll get:

```
RailPulse/
├── README.md            ← this guide
├── requirements.txt     ← Python dependencies
├── setup.bat            ← one-time Windows installer
├── start.bat            ← Windows launcher (double-click to run)
├── start.ps1            ← PowerShell launcher
├── start.sh             ← Linux/macOS launcher
├── stop.bat             ← frees ports 8000/7860 if they get stuck
├── Dockerfile / render.yaml / .gitignore / .dockerignore   ← cloud deployment
├── src/                 ← all application code
├── lib/                 ← Leaflet.js map library (served locally, no CDN)
├── data/                ← synthetic dataset + sample input CSV
├── models/              ← trained model bundle (10 models)
├── results/             ← metrics, ablations, charts
└── deploy/              ← Hugging Face Spaces package + Render guide
```

### 3.2 Windows — the easy way (double-click)

1. **Double-click `setup.bat`** — creates a virtual environment (`.venv`) and installs
   all dependencies. Run this **once**. Takes a few minutes.
2. **Double-click `start.bat`** — starts the app. A console window opens and prints two URLs.
3. Open **http://127.0.0.1:8000** in your browser.

Stop it by pressing **Ctrl+C** in the console (or run `stop.bat` if a port gets stuck).

### 3.3 Windows — manual (Command Prompt / PowerShell)

```bat
cd C:\RailPulse

:: 1. create & activate a virtual environment (one time)
python -m venv .venv
.venv\Scripts\activate

:: 2. install dependencies (one time)
python -m pip install --upgrade pip

pip install numpy pandas scikit-learn networkx matplotlib joblib xgboost lightgbm fastapi uvicorn python-multipart gradio requests
 or
pip install -r requirements.txt

:: 3. run (every time)
python src\run.py
```

### 3.4 Linux / macOS

```bash
cd RailPulse

# one time
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# run
python src/run.py
# or simply:  ./start.sh
```

### 3.5 What `run.py` does

`python src\run.py` launches **two things at once**:

| Service | Port | What it is |
|---------|------|------------|
| **FastAPI backend + web app** | 8000 | The main RailPulse dashboard (login, map, Copilot, prediction) |
| **Gradio console** | 7860 | An older "Deep-Tech Noir" dispatcher console |

> You only need **port 8000** — that's the modern dashboard described in this guide.

If `models/bundle.joblib` is ever missing or out of date, the app **auto-trains** on
first boot (generate data → features → 10 models), which takes 2–3 minutes.

### 3.6 Run each service separately (advanced)

```bash
# Terminal 1 — API + dashboard
cd RailPulse/src
python -m uvicorn app:app --host 0.0.0.0 --port 8000

# Terminal 2 — Gradio console (optional)
cd RailPulse/src
python app_frontend.py
```

---

## 4. Signing in

The app opens on a **login page** (you never land directly on the dashboard).

```
username: admin
password: swr2026
```

You can also create an account via the **Create Account** tab (accounts are stored in
memory for this demo build and reset on restart).

**Change the credentials** with environment variables before launching:

```bat
:: Windows (cmd)
set SWR_USER=dispatcher
set SWR_PASS=my-secret-password
python src\run.py
```

```bash
# Linux / macOS
SWR_USER=dispatcher SWR_PASS=my-secret-password python src/run.py
```

---

## 5. Feature guide — every screen explained

After login you see the sidebar navigation. Here is every screen and what it does.

### 🏠 Dashboard
The landing page — at-a-glance summary cards:
- **Headline result** — graph features cut MAE ~46% vs isolated features.
- **Radius of influence** — the delay ripple decays with distance.
- **What drives delays** — live congestion + weather dominate; graph columns make the model network-aware.
- **Quick start** and a **NetworkX Graph Engine** explainer.

### 🎛️ Prediction (manual entry)
Fill in a live train status and click **⚡ Run Cascade Prediction**:
- **Current station**, **destination**, **upcoming station** (dropdowns populated from the real network)
- **Train type**, **train number**, **weather**, **day of week**, **scheduled hour**, **current delay (min)**

Outputs:
- **Grid Cascade Alert** — colour-coded severity (CRITICAL / HIGH / MEDIUM / LOW) with the projected
  ripple and the top at-risk junction.
- **Route details** — the train's path, distance, stops, and estimated duration.
- **Station-wise delay** — a table projecting the knock-on delay at each downstream stop.
- **Route map** — the train's journey traced on the real map with the at-risk stations shaded.
- **Feature snapshot** — the exact 19 graph-flattened features fed to the model.

### 🤖 Dispatcher Copilot *(the standout feature)*
A chat panel where you **talk to the network in plain English**. It turns your question
into tool calls against the live model + NetworkX graph and answers with real numbers.

Ask things like:
> *"Mysuru delayed 45 min at 18:00 — which junctions choke first and what's my best mitigation?"*

It can: predict delays, rank choke points, find routes, describe stations, summarise the
network, and run **what-if simulations**. Answers with a route/cascade get a **🗺 Show on map**
button. Tap the **🎙 mic** to ask by voice — the reply is read back aloud (🔊 to repeat).
Full details in [§10](#10-the-dispatcher-copilot-ai-agent).

### 🕸️ Network Graph (the map)
The Google-Maps-style interactive railway network — full guide in [§6](#6-the-interactive-map-how-to-use-it).

### 🗂️ Import Dataset (batch prediction)
Upload a CSV of many live train statuses → batch predictions in a table. Each row has a
**🔍 View Prediction** button that opens the full per-train analysis (cascade alert, route
map, station-wise delay). A downloadable **export** link is produced.

CSV columns (only `current_station` is mandatory):
```csv
current_station,train_type,hour,day,weather,current_delay_min,destination,upcoming_station
Mysuru,Superfast,17,Monday,Clear,45,KSR Bengaluru,Mandya
```
A ready-made template is available at `data/sample_input.csv` (and linked from the page).

### 📼 Demo Records
Pre-canned dispatcher scenarios run live through the trained model — a quick way to see
the cascade engine in action.

### 🧠 ML Models
The **10-model bake-off** table (MAE / RMSE / R²):
① DBSCAN (unsupervised delay-sink clustering → feature) · ② Random Forest · ③ Extra Trees ·
④ AdaBoost · ⑤ Gradient Boosting · ⑥ XGBoost · ⑦ LightGBM · ⑧ CatBoost ·
⑨ KNN Bagging · ⑩ **Network-Aware Voting Regressor** (the live model).

### 🔗 NetworkX Graph Engine
Per-station centrality table (degree, eigenvector, PageRank, betweenness) with an
explanation of how the 2-D graph is "flattened" into 1-D columns for classical boosting.

### 🕘 Prediction History
Every prediction you've run, with timestamp, station, destination, predicted delay and alert level.

### 📊 Analytics
- 10-model bake-off chart and the **radius-of-influence** learning curve.
- SHAP-style **feature importance** chart.
- **Ablation studies** table (isolated vs graph-aware, and radius sweep).

### ⚙️ Settings
Active model, network version, feature count, train/test sizes, and credential notes.

### ℹ️ About
The problem statement, the idea, and the proof (the two ablations).

---

## 6. The interactive map — how to use it

The map is **Leaflet.js + real map tiles** (OpenStreetMap in light mode, Esri Dark Gray
Canvas in dark mode) — it is *not* a static image and *not* a fake drawing.

**Controls**
| Action | How |
|--------|-----|
| Zoom in / out | mouse wheel, double-click, pinch, or the **＋ / －** buttons |
| Pan / move | click and drag |
| Reset / fit network | **⤢** button |
| Zoom to a station | **⌖** button (uses the selected station) |
| Fullscreen | **⛶** button |
| Layer control | bottom-left panel (☑ stations, routes, labels, cascade, risk, incident) |
| Base map switch | bottom-left panel (🗺 Light — OpenStreetMap / 🌙 Dark — Esri) |

**Map views** (top toolbar): **Network** (base), **⏱ Live delay**, **⚠ Cascading risk** —
these recolour the whole grid by current delay or choke-risk.

**Delay colour code** (used everywhere — markers, lines, alerts):

| Colour | Meaning |
|--------|---------|
| 🟢 Green | Normal (< 5 min) |
| 🟡 Yellow | Minor delay (5–14 min) |
| 🟠 Orange | Moderate delay (15–29 min) |
| 🔴 Red | Severe delay (30–59 min) |
| ⚫ Dark red | Critical cascade risk (≥ 60 min) |

**Cartography**
- Rail tracks drawn as **roads** — dark casing + lighter fill, width scales with distance.
- Stations are **white-ringed markers** sized/coloured by centrality at their real lat/lon.
- **Green rings** mark the critical hubs (highest betweenness).
- **City labels** with halo; only major stations show at low zoom, more appear as you zoom in.
- Click a **station** → popup card (code, delay, risk, degree, betweenness, PageRank,
  cascade-delay index, grid-congestion score, connected stations).
- Click a **route** → popup (source → destination, km, status).

**Animate Cascade** — picks the source station + delay from the toolbar and replays the
ripple over 3 hops with direction arrows, shaded risk areas, and a live ripple table.
**✕ Clear incident** resets the map.

---

## 7. How the ML works (the science)

### The pipeline

```
data_generator.py        → 10,000 synthetic journeys (NTES-style, coupled delays)
feature_engineering.py   → + graph features + DBSCAN delay clusters  (features.csv)
train.py                 → trains 10 models + ablations + SHAP-style importance
inference.py             → CascadePredictor: live prediction + cascade alert
app.py / copilot.py      → FastAPI backend + AI agent
dashboard.html           → single-file web app (login, theme, Leaflet map, Copilot)
```

### The key idea — "flatten the graph"

NetworkX computes, per station: **degree**, **degree centrality**, **eigenvector
centrality**, **PageRank**, **betweenness**, **closeness**, plus **dynamic congestion
rings** (the live delay of all trains within 1/2/3 edges) and a **cascading delay index**.
These become ordinary tabular columns, so classical gradient boosting becomes
*network-aware* — deployable on modest hardware, no GNN required.

### Results (bundled in `results/`)

- **Best single model:** CatBoost, MAE 4.84 min. **Live dashboard model:** Network-Aware
  Voting Regressor.
- **Ablation 1** (LightGBM): isolated-only MAE 9.52 → full graph-aware **5.09 min (−46%)**.
- **Ablation 2** (radius of influence): r1 → r1+r2 → r1+r2+r3 improves steadily but the
  marginal gain **decays with distance** — the ripple fades.

---

## 8. Reproducing the full pipeline

If you want to regenerate everything from scratch (data → features → models → charts):

```bash
# from the project root
python src/data_generator.py        # 1. synthetic data
python src/feature_engineering.py   # 2. graph features + DBSCAN
python src/train.py                 # 3. train 10 models + ablations
python src/visualization.py         # 4. regenerate figures/ charts
```

> Tip: delete `models/bundle.joblib` first if you want a guaranteed clean retrain.
> CatBoost is optional — the pipeline runs fine without it (9 models + DBSCAN).

---

## 9. API reference

The backend is FastAPI on port 8000.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | the web app (dashboard.html) |
| GET | `/health` | health check (`{"status":"ok",...}`) — used by Render |
| POST | `/login` | validate credentials |
| POST | `/register` | create an account (in-memory) |
| GET | `/stations`, `/stations/{name}` | station list / station metadata |
| GET | `/graph` | full network JSON — nodes (id, code, lat, lon, centralities) + edges |
| GET | `/system/overview` | active model, network version, dataset sizes |
| GET | `/model/metrics` | 10-model bake-off |
| GET | `/history` | recent predictions |
| POST | `/predict/manual` | single prediction (JSON body) |
| POST | `/predict/upload` | CSV upload → batch predictions + export id |
| GET | `/demo/records` | pre-canned scenarios |
| GET | `/exports`, `/export/{id}` | list / download exports |
| GET | `/copilot/status` | Copilot mode (offline / llm) |
| POST | `/copilot/chat` | ask the Copilot a question in natural language |
| GET | `/gps/nearest?lat=&lon=` | nearest station for a GPS coordinate |
| GET | `/gps/route?origin=&destination=` | shortest rail route + coordinates (demo GPS) |
| POST | `/gps/predict` | GPS position → nearest station → prediction → cascade |
| GET | `/models` | selectable model menu (top-5) + active model |
| POST | `/model/select` | switch the active prediction model (`{"model":"CatBoost"}`) |

Example:

```bash
curl -X POST http://127.0.0.1:8000/predict/manual \
  -H "Content-Type: application/json" \
  -d '{"current_station":"Mysuru","destination":"KSR Bengaluru",
       "train_type":"Superfast","hour":18,"day":"Monday",
       "weather":"Clear","current_delay_min":45}'
```

---

## 10. The Dispatcher Copilot (AI agent)

The 🤖 tab is an **agentic AI** over the live network. It works in two modes (auto-selected):

| Mode | How it works | Setup |
|------|--------------|-------|
| **Offline** (default) | Deterministic tool-calling engine — parses your question, runs the right tools (prediction, ripple, choke-ranking, routing, mitigation, what-if), composes a cited answer. | none — works out of the box |
| **LLM** | An OpenAI-compatible model picks the tool and composes the final answer. Every number still comes from the real graph/models (tool-calling), so it never hallucinates network facts. | set env vars below |

**Enable the LLM layer** (works with OpenAI, Groq, Together, or a local Ollama):

```bash
# Linux / macOS
export COPILOT_LLM_KEY=sk-...                       # your API key
export COPILOT_LLM_BASE=https://api.openai.com/v1   # any OpenAI-compatible base
export COPILOT_LLM_MODEL=gpt-4o-mini                # model name
python src/run.py
```

```bat
:: Windows (cmd)
set COPILOT_LLM_KEY=sk-...
set COPILOT_LLM_BASE=https://api.openai.com/v1
set COPILOT_LLM_MODEL=gpt-4o-mini
python src\run.py
```

Example questions:
- "Mysuru delayed 45 min at 18:00 — which junctions choke first and what's my best mitigation?"
- "Route from Chennai Central to Hubballi"
- "Top 5 riskiest stations right now"
- "Tell me about Guntakal"
- "What if I hold Mysuru 20 extra minutes?"

---

## 10a. New — Voice input · live GPS tracking

Two integrated dispatcher features (both reuse the existing graph + ML model — no
separate prediction system):

**🎙 Voice / natural-language input** (Dispatcher Copilot page) — tap the **🎙 mic**
button next to the Copilot's text box and speak a question like *"Mysuru delayed 45
min at 18:00 — which junctions choke first?"*. The browser's Web Speech API
transcribes it, sends it to the Copilot, and the answer is read back aloud — every
bot reply also has a **🔊 Hear reply** button. Uses no key and needs no setup; falls
back to typing if the browser has no speech support.

**🛰 Live Train Tracking** (sidebar) — the full pipeline per train position:

```
GPS (lat,lon) → nearest station → current delay → ML prediction
             → NetworkX cascade analysis → updated risk → map
```

- **Demo GPS mode** (default, for demos): a train moves along the real
  NetworkX shortest route (`/gps/route`) at the speed you set, updating every 5 s.
  A flashing **🛰 DEMO GPS MODE** banner is shown while simulated data is used.
- **Live GPS mode**: uses the device's real geolocation (`navigator.geolocation`).
- The panel shows train number, GPS position, nearest station, next station,
  destination, speed, current delay, predicted destination delay and cascade level.
- Affected stations are coloured **green = low, yellow = medium, red = high/critical**.
- A 0-minute delay is reported and displayed as **0** — never bumped to a minimum.

**🔐 Password & email restrictions** (Create Account) — new accounts now require a
valid email (`name@domain.tld`) and a password with **8+ characters, one uppercase,
one lowercase, one digit and one special character**, validated in both the browser
and the backend. The built-in demo login (`admin` / `swr2026`) is unchanged.

**🧠 Prediction model** — every prediction (manual, batch, GPS tracking) runs on
the **Network-Aware Voting Regressor** (XGBoost + LightGBM + Extra Trees trained on
graph-derived features). The **10-model bake-off** (CatBoost 4.84 min MAE,
GradientBoosting 4.94, NetworkAwareVoting 5.07, LightGBM 5.07, XGBoost 5.11, …)
remains viewable on the **Analytics** page.

**🚆 Train Number is required** — the Delay Prediction and Live Tracking forms both
mark the train number as mandatory and block the request until it is entered.

New endpoints: `GET /gps/nearest`, `GET /gps/route`, `POST /gps/predict`.

---

## 11. Deploying to the cloud (Render)

Get a **permanent public HTTPS link** (free tier).

### Step 1 — push to GitHub

```bat
git init
git add .
git commit -m "RailPulse railway cascade predictor"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/railpulse.git
git push -u origin main
```

### Step 2 — deploy on Render

1. Sign in at https://render.com (with GitHub).
2. **New + → Blueprint** → select your repo.
3. Render reads `render.yaml` and shows the `railpulse` service → click **Apply**.

A few minutes later your app is live at **`https://railpulse.onrender.com`**.

> **Customising:** edit `SWR_USER` / `SWR_PASS` in `render.yaml`, or in
> Render → service → **Environment**. Add `COPILOT_LLM_KEY` there to enable the LLM Copilot.
> Full detail in **`deploy/RENDER.md`**.

### Quick public link without any cloud (temporary)

```bat
:: starts the app + opens a public gradio.live link for the console
python src\run.py --share
```

Or tunnel the main dashboard (port 8000) with Cloudflare Tunnel / ngrok — see `deploy/README.md`.

---

## 12. Running with Docker

```bash
docker build -t railpulse .
docker run -p 8000:8000 -e SWR_USER=admin -e SWR_PASS=swr2026 railpulse
```

Open http://localhost:8000.

---

## 13. Troubleshooting

| Symptom | Likely cause & fix |
|---------|--------------------|
| `'python' is not recognized` | Python not on PATH. Reinstall and tick "Add Python to PATH", then reopen the terminal. |
| `WinError 10048` (port in use) | A previous instance is still running. Close the old console or run `stop.bat`. |
| `ModuleNotFoundError: ...` | Dependencies missing. Re-run `setup.bat` (or `pip install -r requirements.txt`). |
| App opens but map is blank | Your browser has no internet for map tiles (OSM/Esri). The network, markers, popups and cascade still render on the fallback background. |
| "Backend offline" on login | The backend hasn't finished starting (model load ~10–20 s). Wait and refresh. |
| Model retrains on every start | `models/bundle.joblib` is missing or a library version changed. Let it retrain once (2–3 min). |
| Copilot says "backend offline" | Ensure you're running the FastAPI app (port 8000), not just the Gradio console. |

---

## 14. Project structure

```
RailPulse/
├── README.md                ← this guide
├── requirements.txt         ← Python dependencies
├── setup.bat / start.bat / start.ps1 / start.sh / stop.bat   ← launchers
├── Dockerfile / render.yaml / .gitignore / .dockerignore      ← cloud deployment
├── src/
│   ├── config.py            ← stations, edges, encodings, paths, constants
│   ├── graph_utils.py       ← NetworkX: graph, centralities, ripple, cascade alert
│   ├── data_generator.py    ← Phase 1: synthetic coupled-delay dataset
│   ├── feature_engineering.py ← Phase 2: graph features + DBSCAN
│   ├── train.py             ← Phase 3–4: 10 models + ablations + SHAP-style importance
│   ├── visualization.py     ← charts in results/figures/
│   ├── custom_models.py     ← KNN-bagging baseline (pickle-safe)
│   ├── inference.py         ← CascadePredictor (shared prediction engine)
│   ├── copilot.py           ← Dispatcher Copilot (agentic AI)
│   ├── app.py               ← FastAPI backend + dashboard server (port 8000)
│   ├── app_frontend.py      ← Gradio console (port 7860)
│   └── run.py               ← launches backend + console together
├── lib/                     ← Leaflet.js + map assets (served locally)
├── data/
│   ├── raw/swr_journeys.csv          ← 10,000 simulated journeys
│   ├── processed/features.csv        ← 31 graph-aware feature columns
│   └── sample_input.csv              ← template for the Dataset tab
├── models/bundle.joblib     ← trained graph + voting regressor + encoders
├── results/
│   ├── model_metrics.csv / ablation_results.csv / feature_importance.csv
│   └── figures/             ← network map, ripple heatmap, bake-off, curves
└── deploy/                  ← Hugging Face Spaces package + Render guide
```

---

**RailPulse** — *Smarter Predictions. Smoother Journeys.*
