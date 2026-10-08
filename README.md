# 🚄 RailPulse — Railway Cascade Prediction System

**Graph-Derived Feature Boosting for Cascading Train Delay Prediction on Indian Railways**

RailPulse is an end-to-end, ML-powered **dispatcher workstation** for Indian Railways
(centred on the South Western Railway zone, extended to a 84-station national network).
It predicts how a delay at one station *cascades* through the rail network, ranks the
junctions most likely to choke, recommends mitigations, and lets you talk to the network
in plain English through a built-in **AI Dispatcher Copilot** — all on a real,
Google-Maps-style interactive map.

> **Headline result:** graph-derived features cut delay-prediction error (**MAE**) by
> **46.5 %** versus isolated features alone (9.573 → 5.119 min), and most of that gain
> comes from the *dynamic* network state — the delay observed at neighbouring stations —
> rather than from static topology. The radius-of-influence ablation shows the 1-edge ring
> carries nearly all of the signal: adding the 2nd and 3rd rings improves MAE only
> marginally, and this single split does **not** establish that the ripple demonstrably
> decays with distance (see *Results* below for the exact numbers and the caveat).
> *No train is an island* — that part is measured.

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

### How authentication actually works

Authentication is enforced **by the server**, not by hiding the dashboard in the
browser. Signing in returns a signed session token (HMAC-SHA256, 12-hour life);
every subsequent request carries it, and each endpoint declares the role it
requires.

| Role | May do |
|------|--------|
| `viewer` | read the network, metrics, overview |
| `dispatcher` | + run predictions, exports, Copilot |
| `controller` | + switch the active model, ingest real data |
| `admin` | + read the audit trail, manage users and roles |

Practical consequences:

* **Passwords are hashed** with PBKDF2-HMAC-SHA256 (280,000 iterations, per-user
  salt) and stored in `data/users.json` — plaintext passwords are never written.
  The file is git-ignored.
* **Brute force is throttled**: five failed attempts lock an account/IP pair out
  for five minutes, and each failure is recorded in the audit trail.
* **A never-committed signing key** is generated on first boot into
  `data/.auth_secret` (or supplied via `SWR_SECRET_KEY`).
* **Accounts created via "Create Account"** get the lowest operational role
  (`dispatcher`), never admin. Self-registration can be switched off entirely
  with `SWR_ALLOW_SELF_REGISTRATION=0`.

You can verify the whole model yourself:

```bash
python -m pytest tests/test_auth.py tests/test_api.py -v
```

> ⚠️ The published demo account (`admin` / `swr2026`) exists so the offline demo
> works out of the box. The server logs a warning on every boot while it is in
> use — set `SWR_USER`/`SWR_PASS` before any real deployment.

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
                           (+ any ingested real observations, with a provenance record)
train.py                 → 10 models + ablations + SHAP-style importance
                           + quantile models + conformal calibration
inference.py             → CascadePredictor: prediction + interval + cascade alert
uncertainty.py           → quantile regressors, CQR calibration, coverage metrics
sources/                 → real-data adapters: schema mapping, validation, causes, parity
auth.py / audit.py       → server-side sessions, roles, append-only audit trail
model_io.py              → portable model persistence (no version-coupled pickles)
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

Full numbers live in `results/model_metrics.csv`, `ablation_results.csv` and
`feature_importance.csv`; the figures below are read from those files.

- **Best single model:** CatBoost, MAE **4.85 min** (R² 0.860). **Live dashboard model:**
  Network-Aware Voting Regressor — XGBoost + LightGBM + Extra Trees over graph features.
- **Ablation 1** (LightGBM, nested feature sets): isolated-only MAE **9.573** → +static
  graph 7.834 → +dynamic network 5.730 → full graph-aware **5.119 min (−46.5 %)**. Most of
  the gain comes from the *dynamic* network state, not the static topology.
- **Ablation 2** (radius of influence): r1 5.390 → r1+r2 5.283 → r1+r2+r3 **5.140**.
  Both extra rings help slightly, but the third ring's marginal gain (0.143 min) is **not
  smaller** than the second's (0.107 min) on this split — with one train/test split and
  ~2,000 test rows, that ordering is within noise. So the data supports *"the 1-edge ring
  carries almost all of the signal"* but **not** a claim that the ripple demonstrably decays
  with distance; that would need repeated splits to establish.

### Uncertainty: you get an interval, not just a number

A point estimate cannot answer *"will it miss the connection?"* — that is a
threshold question. RailPulse therefore reports **conformalized quantile
regression** intervals (Romano, Patterson & Candès, 2019):

1. Nine LightGBM quantile regressors (τ = 0.025 … 0.975) model the conditional
   delay distribution, not just its centre.
2. A **held-out calibration split** (10 % of the data, never used for fitting)
   provides the conformity scores `max(q_lo − y, y − q_hi)`.
3. Each interval is widened by the finite-sample quantile of those scores, which
   gives the interval a **coverage guarantee** on future data.

Measured on the 2,000-row test split (see `results/uncertainty_metrics.csv`):

| Nominal | Raw quantile coverage | Conformal (CQR) coverage |
|---------|----------------------|--------------------------|
| 50 % | 35.9 % ❌ | **51.6 %** ✅ |
| 80 % | 61.5 % ❌ | **82.7 %** ✅ |
| 90 % | 75.8 % ❌ | **90.5 %** ✅ |
| 95 % | 79.8 % ❌ | **94.8 %** ✅ |

Conformal calibration cuts the mean coverage shortfall by **93 %**, and the
reliability diagram (`results/figures/C1_uncertainty_reliability.png`) plots the
whole calibration curve against the diagonal. Predictions also carry
**exceedance probabilities** — `P(delay > 15 / 30 / 60 min)` — derived by
inverting the predicted quantile function, i.e. the connection-miss risk a
controller actually acts on. Nine quantiles cannot resolve a tail probability of
0.999: estimates outside the trained range are clamped to ±0.5 % and named in
`saturated_thresholds`, so the API says *"≥ 0.995, and here is the limit of what I
can tell you"* instead of printing `0.99` for every threshold. The response also
reports whether the point estimate sits inside the calibrated band
(`consistency.point_inside_model_band`) and, if the band had to be widened to
contain it, gives the unadjusted bounds alongside.

### Which trains does this delay actually affect?

A forecast answers "how late will this train be?". The dispatcher's next question
is *"and who else does that hold up?"* — so RailPulse reconstructs a **timetable**
from the journey record and reasons over it with explicit operating rules.

```bash
curl -s -X POST http://127.0.0.1:8000/cascade/trains \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"current_station":"Mysuru","destination":"KSR Bengaluru",
       "train_type":"Superfast","hour":18,"day":"Monday",
       "current_delay_min":45,"horizon_min":120}' | python -m json.tool
```

```json
{
  "delayed_train": {"at": "Mysuru", "destination": "KSR Bengaluru",
                    "current_delay_min": 45, "predicted_destination_delay_min": 78,
                    "arrives_destination_hour": "21:48"},
  "affected_count": 8,
  "by_severity": {"severe": 0, "moderate": 4, "minor": 4},
  "by_kind": {"FOLLOWING_BLOCK": 7, "PLATFORM_REGULATION": 1},
  "horizon": {"minutes": 120, "analysed_from": "18:40", "analysed_until": "20:45",
              "day_name": "Monday", "candidates_considered": 40},
  "affected_trains": [
    {"train_id": "TRN07926", "train_type": "Superfast", "kind": "FOLLOWING_BLOCK",
     "where": "Maddur → Channapatna", "when": "19:50",
     "expected_added_delay_min": 14.2, "p_over_15min": 0.52, "severity": "moderate",
     "reason": "TRN07926 trails the delayed train into the Maddur–Channapatna block and cannot enter until it clears (+3 min headway).",
     "assumed": false}
  ]
}
```

Four conflict classes, in decreasing confidence:

| Kind | Rule | Depends on |
|------|------|-----------|
| `FOLLOWING_BLOCK` | B cannot enter a section until the delayed train clears it, plus headway | timetable + headway |
| `PLATFORM_REGULATION` | both trains reach the same station inside a platform window | timetable + berth window |
| `RAKE_TURNAROUND` | B departs from where the delayed train terminates, inside the turn-around buffer | timetable + stock rotation |
| `MEET_REGULATION` | head-on crossing inside one section | **a single-line section — assumed, and off by default** |

The delay each affected train absorbs is a **distribution**, not a number: the
delayed train's calibrated quantiles are propagated to every downstream station
and the hold is evaluated per sample, so a row can read "+14 min,
P(> 15 min) = 52%" — which is what a regulation decision actually needs.

**What this claim rests on, stated plainly.** The dataset stores one snapshot per
train, so the timetable is *reconstructed*: paths come from network routing, and
run times from assumed commercial speeds per service type
(`src/timetable.py`, reported in every response). The operating rules (3-min
headway, 5-min platform window, 25-min turn-around buffer, 60 % of a hold
assumed recovered downstream) are stated assumptions in `src/conflicts.py`.
Other trains are assumed to run to their scheduled times. The mechanism is real
railway logic; the magnitudes are only as good as the simulated schedule they run
on — so treat this as a demonstration of the method, not a punctuality forecast
for South Western Railway. Meets are excluded unless you pass
`include_meets: true`, and are flagged `assumed` when you do, because the network
carries no single-/double-line attribute.

### Data provenance — what the model was really trained on

No live NTES feed is openly available (NTES has no public API), so the published
model is trained on a **simulated** delay field. Rather than leave that in a
footnote, it is recorded in the model bundle and surfaced as a badge in the
dashboard header; `/system/overview` reports it programmatically.

The moment real data is available, `src/sources/` ingests it. A batch only has to
identify **three** things — where the train is, how late it is, and how late it
reached its destination (the label). Everything else is derived from the network
graph or imputed with a reported default, so a sparse operator export is usable:

```csv
Station,Train No,Delay,Arrival Delay,To Station,Wx,Sch Time,Day
Mysuru,12627,45,95,Dharwad,fog,14:05,Mon
```

A batch with **no** label is refused rather than stored: a live position feed is
not training data, and scoring it is what `/predict/*` and `/gps/*` are for.

| Step | What happens |
|------|--------------|
| **Map** | Operator dialect → canonical schema (`Station`→`current_station`, `Sch Time`→`scheduled_hour`, …) with an override map for unusual exports. A column claimed by two fields (e.g. `Arrival Delay`) is resolved deterministically **and reported** so you can override it |
| **Normalise** | Station codes/names resolved to graph nodes; hours/days parsed from `14:05`, `7 PM`, `Mon`; missing fields derived from the graph, defaults reported as **imputed**, and a missing delay **never** invented |
| **Validate** | Railway-plausible bounds (delay ≤ 24 h, hour 0–23, section resolvable and plausible, placement sane, duplicates, completeness) — rejected rows are itemised with the reason, never silently dropped |
| **Attribute** | Delay **cause heads** (pre-occupied line, rolling stock, engineering/TRT, crew/HOER, signalling, natural, convention) with the department that owns each fix |
| **Check parity** | Two-sample **KS test** against the training distribution: *is this data even from the world the model learned?* A drift verdict says the reported metrics cannot be expected to transfer |
| **Record** | Provenance upgrades to `synthetic+real` or `real`, and travels into the next trained bundle |

```bash
# ingest an operator export, then look at what the causes cost you
curl -X POST http://127.0.0.1:8000/ingest/upload -H "Authorization: Bearer $TOKEN" \
     -F "file=@swr_delay_export.csv"
curl -s http://127.0.0.1:8000/data/causes  -H "Authorization: Bearer $TOKEN"
curl -s http://127.0.0.1:8000/data/parity  -H "Authorization: Bearer $TOKEN"
```

### Model persistence that survives upgrades

Estimators are stored in each library's **own** format (XGBoost `.json`,
LightGBM `.txt`, CatBoost `.cbm`) instead of being pickled, and the dashboard's
fused ensemble is rebuilt from its components. `train.py` asserts that the
rebuilt ensemble reproduces the original to within `1e-4` minutes before it will
publish the bundle (measured difference: **0.0**), and `tests/test_model_io.py`
re-checks it. Previously the server warned on every boot that the pickled
XGBoost model came from an older version — that class of silent model drift is
gone.

---

## 8. Reproducing the full pipeline

If you want to regenerate everything from scratch (data → features → models → charts):

```bash
# from the project root
python src/data_generator.py        # 1. synthetic data
python src/feature_engineering.py   # 2. graph features + DBSCAN
python src/train.py                 # 3. train 10 models + ablations + conformal calibration
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
| POST | `/login` | exchange credentials for a signed session token |
| POST | `/register` | create an account (PBKDF2-hashed; lowest role) |
| GET | `/stations`, `/stations/{name}` | station list / station metadata |
| GET | `/graph` | full network JSON — nodes (id, code, lat, lon, centralities) + edges |
| GET | `/system/overview` | active model, network version, dataset sizes |
| GET | `/model/metrics` | 10-model bake-off |
| GET | `/history` | recent predictions |
| POST | `/predict/manual` | single prediction (JSON body) |
| POST | `/cascade/trains` | which upcoming trains a delay affects, with expected added delay |
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

### Authentication, governance and real-data endpoints (new)

| Method | Endpoint | Min role | Purpose |
|--------|----------|----------|---------|
| GET | `/auth/config` | public | what auth this deployment expects (no secrets) |
| GET | `/auth/me` | viewer | validate a token; returns identity + permissions |
| POST | `/auth/logout` | viewer | ends the session (recorded in the audit trail) |
| GET | `/admin/audit` | admin | the append-only audit trail (`?limit=&event=&username=`) |
| GET | `/admin/users` | admin | list accounts and roles |
| POST | `/admin/users/{username}/role` | admin | change a role (`{"role":"controller"}`) |
| POST | `/ingest/upload` | controller | ingest an operator export (CSV/Excel) |
| POST | `/ingest/records` | controller | live-feed webhook (≤ 5,000 JSON records per call) |
| GET | `/ingest/status` | viewer | what has been ingested + resulting provenance |
| GET | `/data/causes` | viewer | delay-minutes per cause head, with escalation owner |
| GET | `/data/parity` | viewer | KS drift test: is the new data like the training world? |

Every operational endpoint returns **401** without a token and **403** if the
role is insufficient, so the API can be tested for that property directly (see
`tests/test_api.py`).

The token is accepted over **four carriers** — `Authorization: Bearer`, an
`X-RailPulse-Token` header, the `HttpOnly` session cookie, or a `?token=` retry —
and any one of them is sufficient. Hosting layers differ: a proxy can strip
`Authorization`, and a browser refuses a third-party cookie inside an embedded
frame, either of which produces the confusing "signed in, then 401 everywhere".
The cookie is issued `SameSite=None; Secure` over HTTPS (survives a frame) and
`SameSite=Lax` over plain HTTP for local development. `GET /auth/diagnostics`
reports which carriers the server actually received for the caller's own request
(booleans only), and a 401 names the missing or rejected credential — that reason
is also written to the audit trail as `auth_rejected`.

Sign-in and data loading are separate steps on the dashboard: only the `/login`
exchange can report an access-denied verdict, so a failure to load data is never
announced as bad credentials.

Nothing on the dashboard may call an authenticated endpoint before sign-in —
`tests/boot_probe.mjs` executes the page's real JavaScript in Node and fails if
any protected route is requested without a session.

Example — the full authenticated flow:

```bash
# 1. sign in and capture the token
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"swr2026"}' | python -c 'import sys,json;print(json.load(sys.stdin)["token"])')

# 2. predict (note the conformal interval and exceedance probabilities)
curl -s -X POST http://127.0.0.1:8000/predict/manual \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"current_station":"Mysuru","destination":"KSR Bengaluru",
       "train_type":"Superfast","hour":18,"day":"Monday",
       "weather":"Clear","current_delay_min":45}' | python -m json.tool

# 3. read the audit trail
curl -s "http://127.0.0.1:8000/admin/audit?limit=10" \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

A prediction response now includes the uncertainty block:

```json
"uncertainty": {
  "available": true,
  "method": "conformalized quantile regression (CQR)",
  "median": 47.3,
  "intervals": {
    "50": {"lower": 38.1, "upper": 52.0, "width": 13.9},
    "80": {"lower": 31.4, "upper": 58.8, "width": 27.4},
    "90": {"lower": 27.9, "upper": 63.2, "width": 35.3}
  },
  "exceedance": {"p_gt_15min": 0.94, "p_gt_30min": 0.71, "p_gt_60min": 0.18}
}
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
graph-derived features). The **10-model bake-off** (CatBoost 4.85, GradientBoosting 5.02, NetworkAwareVoting 5.08, LightGBM 5.14, XGBoost 5.08 min MAE, …)
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
