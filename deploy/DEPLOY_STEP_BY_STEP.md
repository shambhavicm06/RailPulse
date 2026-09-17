# 🚀 RailPulse — Deploy Step-by-Step (Render, free)

A **complete, click-by-click** guide. Two situations are covered:

* **Part A** — you already deployed once (you have a GitHub repo + a Render
  service). You just changed code and want the live site to update.
* **Part B** — you are starting from zero (no GitHub repo, no Render account).

At the end: **Part C** runs it locally for a demo/viva, and **Part D** is a
verification checklist + troubleshooting.

> Your current setup (verified): GitHub repo `shambhavicm06/TrainPulse`,
> Render service `railpulse` with `autoDeploy: true`. So **Part A** is what you
> need right now.

---

## What you need (one time)

| Thing | Where | Notes |
|---|---|---|
| GitHub account | https://github.com | free |
| Render account | https://render.com | sign in **with your GitHub login** (one click) |
| Git on your PC | https://git-scm.com/download/win | default install is fine |

No credit card. Free tier on both.

---

# ✅ PART A — Update your existing deployment (what you need now)

Your Render service watches the GitHub repo. When you **push new code**, Render
rebuilds and redeploys **automatically** (because `render.yaml` has
`autoDeploy: true`). So the whole thing is just: push to GitHub.

### A1. Open a terminal in the project folder
On Windows, open the folder `railway-cascade-predictor` in File Explorer,
click the address bar, type `cmd`, press **Enter**. (Or use Git Bash.)

### A2. Stage + commit + push
Run these, one line at a time:

```bat
git add .
git commit -m "Restore UI + ML Models menu + same-origin tile proxy"
git push
```

*If it asks for a login*, use your GitHub username and a **Personal Access
Token** (GitHub → Settings → Developer settings → Personal access tokens →
Generate new token → tick `repo`). Paste the token as the password.

### A3. Watch Render rebuild
1. Open https://dashboard.render.com
2. Click your service **railpulse**.
3. You'll see a new deploy start automatically ("Building…"). Wait 3–6 minutes.
4. When it says **Live**, open your app URL (top of the page,
   `https://railpulse-xxxx.onrender.com`) and sign in `admin` / `swr2026`.

That's it — the live site now has the restored UI, the 5-model dropdowns and the
fixed map. **No Render settings need to change.**

> 💡 The first visit after a quiet period "wakes" the free instance (30–60 s).
> If the login briefly says *Backend offline*, wait ~30 s and refresh.

---

# 🆕 PART B — Fresh deploy from scratch

### B1. Put the code on GitHub
In the project folder (terminal):

```bat
git init
git add .
git commit -m "RailPulse railway cascade predictor"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<REPO>.git
git push -u origin main
```

(On GitHub, first create an **empty** repo named `<REPO>` — do *not* add a
README so the push isn't blocked.)

### B2. Connect Render to GitHub
1. Go to https://render.com → **Get Started** → **GitHub** (sign in).
2. Render asks to authorise GitHub → allow.

### B3. Create the service from the Blueprint
1. Render Dashboard → **New +** → **Blueprint**.
2. Pick your repo (`<YOUR_USERNAME>/<REPO>`).
3. Render reads `render.yaml` and shows one web service **railpulse** (Free).
4. Click **Apply**.

Render now builds the Docker image (~4–8 min first time) and deploys.

### B4. (Alternative) Manual Web Service
If you prefer not to use the Blueprint:
1. **New +** → **Web Service** → pick your repo.
2. Render auto-detects the `Dockerfile`.
3. Name: `railpulse` · Instance: **Free**.
4. **Advanced** → Health Check Path: `/health`
   → add env vars `SWR_USER=admin`, `SWR_PASS=swr2026`.
5. **Create Web Service**.

### B5. Get your URL + sign in
1. When the deploy is **Live**, open the URL at the top
   (`https://<name>.onrender.com`).
2. Sign in with **`admin` / `swr2026`** (or your `SWR_USER`/`SWR_PASS`).

---

# 💻 PART C — Run locally (for a demo / viva, no internet needed for the app)

From the project folder:

```bat
start.bat        :: installs deps (first time) and starts the server
```

or by hand:

```bat
pip install -r requirements.txt
python src/run.py
```

Then open **http://localhost:8000** and sign in `admin` / `swr2026`.

---

# 🔍 PART D — Verify it worked

Open these (replace `<URL>` with your Render URL):

| Check | URL | Expect |
|---|---|---|
| Health | `<URL>/health` | `{"status":"ok",...}` |
| Login page | `<URL>/` | login form |
| Tiles (map) | `<URL>/tiles/osm/5/24/15.png` | a real map image (not an error) |
| Models | `<URL>/models` | list of 5 models |

Then in the app: run a prediction (needs a Train Number) and confirm the
**Route Map** shows real tiles and the result card shows a predicted delay.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Build fails on `pip install` | Usually transient — **Manual Deploy → Deploy latest commit** |
| "Backend offline" at login | Instance still waking — wait ~30 s, refresh |
| Map tiles slow first time | The proxy caches tiles; first load fetches, later loads are fast |
| Push rejected (remote has work) | `git pull --rebase` then `git push` |
| Forgot password | It's `SWR_USER`/`SWR_PASS` in Render → service → **Environment** |

## Optional: turn on the Copilot's LLM layer
Render → your service → **Environment**, add:
* `COPILOT_LLM_KEY` = your API key
* `COPILOT_LLM_BASE` = e.g. `https://api.openai.com/v1`
* `COPILOT_LLM_MODEL` = e.g. `gpt-4o-mini`

Without these the Copilot still works in offline (deterministic) mode.

---

*The map now proxies tiles through the app server (`/tiles/...`), so OpenStreetMap
"Access blocked" tiles can never appear on Render or in any preview.*
