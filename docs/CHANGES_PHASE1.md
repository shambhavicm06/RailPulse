# Phase 1 — "Credit": what changed and how to verify it

Goal: make every claim the app makes either **true**, **measurable**, or
**labelled**. Nothing here changes the research spine (graph-derived features →
10-model bake-off → ablation); it changes what the system is allowed to claim
about itself.

Everything below can be reproduced from a clean checkout.

---

## 1. The API is authenticated (it was open before)

**Before:** `grep -c "Depends\|verify_token\|require_auth" src/app.py` → **0**.
`/login` returned `{"success": true}` and `dashboard.html` simply hid the
dashboard. Every endpoint — predictions, exports, Copilot — was reachable by
anyone with the URL.

**Now:** sessions are signed tokens, and each endpoint declares the role it needs.

| Role | Capability |
|------|-----------|
| `viewer` | read the network, metrics, overview |
| `dispatcher` | + predictions, exports, Copilot |
| `controller` | + switch the active model, ingest data |
| `admin` | + audit trail, user/role administration |

* Passwords: PBKDF2-HMAC-SHA256, 280,000 iterations, per-user salt
  (`data/users.json`, git-ignored — plaintext is never stored).
* Tokens: HMAC-SHA256 over a JSON payload with an explicit expiry (12 h);
  tampering invalidates the signature.
* Login throttling: 5 failures / 5 min per username+IP, `429` with `Retry-After`.
* No user enumeration: an unknown username answers exactly like a wrong password.
* CORS changed from `allow_origins=["*"]` to an explicit allow-list.
* Signing key: `data/.auth_secret` (0600) or `SWR_SECRET_KEY`; never committed.

**Verify**

```bash
python -m pytest tests/test_auth.py tests/test_api.py -v
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/graph      # 401
```

---

## 2. Predictions come with calibrated uncertainty

**Before:** MAE/RMSE/R² only. A controller cannot act on "48 minutes" —
they need *"P(miss the 15-minute connection) = ?"*.

**Now:** nine LightGBM quantile regressors (τ = 0.025 … 0.975) plus
**conformalized quantile regression** (Romano, Patterson & Candès, 2019) using a
held-out calibration split that no model is fitted on.

Measured on 2,000 held-out journeys:

| Nominal | Raw quantile coverage | Conformal (CQR) coverage |
|---------|----------------------|--------------------------|
| 50 % | 35.1 % | **51.5 %** |
| 80 % | 62.9 % | **82.3 %** |
| 90 % | 75.9 % | **90.4 %** |
| 95 % | 79.7 % | **94.9 %** |

Mean coverage shortfall: **+0.153 → −0.010** (93 % reduction). Reliability
diagram: `results/figures/C1_uncertainty_reliability.png`; numbers in
`results/uncertainty_metrics.csv`.

The API returns the band **plus** exceedance probabilities
`P(delay > 15 / 30 / 60 min)`, computed by inverting the predicted quantile
function. Estimates outside the trained quantile range are clamped to ±0.5 % and
listed in `saturated_thresholds` rather than reported as a confident 0.99.

**Verify**

```bash
python -m pytest tests/test_uncertainty.py tests/test_api.py -v
curl -s -X POST http://127.0.0.1:8000/predict/manual -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"current_station":"Mysuru","destination":"KSR Bengaluru","current_delay_min":45}' \
  | python -m json.tool | grep -A20 uncertainty
```

---

## 3. Real data can be ingested, validated and checked for drift

**Before:** the pipeline could only read its own generator output.

**Now:** `src/sources/` provides the complete path from an operator export to a
model-ready record:

| Stage | Module | What it guarantees |
|-------|--------|--------------------|
| Column mapping | `sources/base.py` | operator dialects (`Station`, `Sch Time`, `To Station`, …) → canonical schema; unmapped columns are **reported** |
| Normalisation | `sources/base.py` | station names/codes → graph nodes; hours from `14:05`/`7 PM`; congestion rings derived and **labelled as a proxy** |
| Validation | `sources/validate.py` | railway-plausible bounds (delay ≤ 24 h, hour 0–23, duplicates, completeness); rejections itemised |
| Cause attribution | `sources/causes.py` | delay heads (pre-occupied line, rolling stock, engineering/TRT, crew/HOER, signalling, natural, convention) + the department owning each fix |
| Parity | `sources/parity.py` | two-sample **KS test**: does the new data come from the world the model learned? |
| Provenance | `sources/ingest.py` | upgrades to `synthetic+real` / `real` and travels into the next bundle |

**Verify**

```bash
python -m pytest tests/test_pipeline.py -v
```

---

## 4. Nine genuine bugs found by probing the running system

All were pre-existing defects, found by comparing what the model *learned* with
what the app *served*, and by driving the running system instead of trusting it.
Each one is now covered by a regression test or an explicit assertion.

### 4.1 `delay_cluster` was fed as `0` at inference

Training labels come from DBSCAN (values **0–42**, where 0 means "noise").
Inference derived a proxy from eigenvector centrality
(`min(4, int(ev * 10))`) which evaluates to **0 for essentially every station** —
so the served model received a feature whose value had a *different meaning* from
the one it was fitted on.

*Fix:* `feature_engineering.py` persists the per-station label map
(`data/processed/station_clusters.json`); it travels in the bundle; inference
reads the real label. Verified: **0 of 84 stations disagree**.
Measured effect before the fix: −0.8 to −1.7 min per prediction.

### 4.2 The congestion fallback was outside the training distribution

With no live feed, inference synthesised `congestion_r1 = 0.5 × delay`. The
training data has a median ratio of **1.04** (0.51–3.31 across stations) — the
model's **second-strongest feature** was being fed at roughly half scale.

*Fix:* `train.py` fits `congestion_feature ≈ slope × delay + intercept`
**per station** on the training sample and stores it in the bundle.
For delay = 45 min at Mysuru:

| Feature | Old proxy | New fallback | Training median |
|---------|-----------|--------------|-----------------|
| `congestion_r1` | 22.5 | 30.9 | 30.2 |
| `congestion_r2` | — | 20.7 | 21.1 |
| `congestion_r3` | — | 15.4 | 15.2 |
| `cascading_delay_index` | 118 | 138.6 | 136.7 |

### 4.3 The ingestion path fabricated its own training labels

`normalise_frame` filled a **missing** delay column with `0.0`:

```python
for column in ("current_delay_min", "destination_arrival_delay_min"):
    ...
    else:
        out[column] = 0.0          # ← "arrived exactly on time"
        derived.append(column)
```

A feed without an arrival-delay column was therefore accepted, and every row was
recorded as a delay of zero — a fabricated supervised label, reported to the user
as *derived* data, that would then be trained on. No check could catch it, because
`0.0` is a perfectly legal delay.

*Fix:* delays are never invented. A missing column is left missing, marked
`unusable`, and the rows are rejected by validation; a batch with no label column
at all is refused outright with a message explaining that a live position feed is
not training data and pointing at `/predict/*` for scoring instead. The report now
separates **derived** (computed from the graph) from **imputed** (a default was
assumed) so an assumption can never be presented as a measurement.

### 4.4 Appending a second feed corrupted the CSV

`DataFrame.to_csv(mode="a")` writes values in its own column order and ignores the
existing header. Two feeds that mapped to different column orders therefore wrote
their values *under the wrong column names* — the stored file looked fine and the
numbers were shifted. Reproduced in this workspace: a second ingest produced rows
like `12627,Superfast,KSR Bengaluru,…` beneath a `train_id,current_station,…`
header.

*Fix:* appends are reindexed to the existing header, and a differing column set
triggers a rewrite with a union header. The tests now run against a temporary
store (`tests/conftest.py`) so the suite cannot pollute or depend on the real one.

### 4.5 The speed check could never fire, yet always reported success

```python
impossible = (km > MAX_PLAUSIBLE_SPEED_KMH) & (km <= 0)   # always False
```

The data-quality gate advertised "speed plausibility … > 160 km/h rejected". The
condition is unsatisfiable (`km > 160 and km <= 0`), so the check never rejected
anything — and always reported *"no physically impossible section timings
detected"*, i.e. it emitted a false assurance about data it had not examined.

*Fix:* replaced with a check that can actually fail — every row must resolve to a
real adjacent-station pair with a plausible section length (the longest section in
the bundled network is 660 km), and rows that resolve no section are rejected
rather than silently handed to the model as a 0 km hop.

### 4.6 The ingestion gate rejected usable feeds

`ingest_dataframe` refused any upload that did not carry all ten original
"required" columns — including `upcoming_station`, `destination` and `origin`,
which `normalise_frame` itself derives from the network graph. A real operator
export with a station, a delay and an arrival delay was therefore rejected by a
pipeline that could have filled in the rest.

*Fix:* the schema now says what it means — **required** (current station, current
delay, destination arrival delay), **derivable** (network-derived) and
**imputable** (defaulted, with the default reported). A minimal four-column
export now ingests end-to-end. A synonym claimed by two fields (`"Arrival
Delay"` plausibly means either the current or the destination delay) is resolved
deterministically *and reported* under `ambiguous_source_columns`, so the operator
can settle it with an override rather than discovering it in the model later.

### 4.7 The dashboard lost its token and reported the session as expired

The signed-in UI showed **"Session expired or not authorised"** on every panel.
The token was stored *only* in `sessionStorage` behind a `catch {}`; in an
embedded/private context where storage access throws, the write failed silently,
login appeared to succeed, and every subsequent request went out unauthenticated.

*Fix:* the token now lives in memory for the session, with `sessionStorage` used
only as a best-effort restore path, and login refuses to continue without a token
from the server. The dashboard is also served `Cache-Control: no-store` — a cached
copy of a pre-auth UI is indistinguishable from a broken login, because it calls
the API the old way.

### 4.8 The dashboard could sign in and still have no credential

Observed on the live preview: `POST /login` → **200**, then every dashboard call →
**401**. Two independent causes can produce that signature — the page was forbidden
from keeping the token (blocked `sessionStorage` in an embedded frame), or the
`Authorization` header never reached the server (a reverse proxy normalising
headers; the app cannot know which hostname a sandbox preview will be served on).

Rather than guess, the token now travels over **two independent transports** and
either one is sufficient:

* the `Authorization: Bearer` header (unchanged), and
* an `HttpOnly`, `SameSite=Lax` session cookie set by `/login` and cleared by
  `/auth/logout`.

The token is also held in memory for the page's lifetime with `sessionStorage`
used only as a best-effort restore path, so a frame that denies storage access can
still authenticate. Requests are sent with `credentials: "include"`, and a
small middleware reflects the request `Origin` **only when its host equals the
host the request was addressed to** — which is what makes an opaque-origin frame's
own API calls work, while granting a foreign origin nothing (verified by test).

Finally, a 401 now says *which* credential was missing or rejected
(`"No session credential arrived…"` vs `"…in Authorization header is invalid"`),
that reason is written to the audit trail as `auth_rejected`, and the dashboard
shows the server's own message instead of a generic "session expired". A failure
that used to be indistinguishable from a bug is now a one-line diagnosis.

### 4.9 A filename collision silently served one model for nine quantile levels

`Path.with_suffix` **replaces** the last suffix, so `q0.025` + `.txt` became
`q0.txt`: all nine quantile models were written to one file, and serving loaded
that single file for every level. The in-memory metrics looked correct while the
served models were wrong — the most dangerous class of bug in this project.

*Fix:* extensions are appended, not substituted (`model_io._target_path`), stems
are dot-free (`q0025.txt` … `q0975.txt`), and `train.py` now **reloads every
artefact it publishes** and refuses to save the bundle if the reloaded model does
not reproduce the fitted predictions (same idea as the ensemble equivalence
check). Regression tests: `tests/test_model_io.py`.

---

## 5. Model persistence no longer breaks on upgrade

**Before:** the server printed this on every boot —

> `WARNING: If you are loading a serialized model ... generated by an older version
> of XGBoost, please export the model by calling Booster.save_model ...`

**Now:** each library's own format (XGBoost `.json`, LightGBM `.txt`, CatBoost
`.cbm`; scikit-learn falls back to joblib with pinned versions), and the fused
ensemble is rebuilt from its components. `train.py` asserts the rebuilt ensemble
reproduces the original `VotingRegressor` to within 1e-4 min — measured
difference **0.000000** — and the reload check above covers the quantiles. The
boot warning is gone and `bundle.joblib` dropped from 30 MB to ~5 KB since it now
holds a graph and manifests rather than pickled estimators.

---

## 6. Silent auto-retrain is gone

`ensure_bundle()` used to retrain at import time whenever a version string
changed — in production that silently swaps the live model mid-shift. Now:

* missing/corrupt bundle → regenerate (unavoidable), logged loudly;
* **topology change** → reported, and retrained **only** with
  `SWR_AUTO_RETRAIN=1`; otherwise the existing model keeps serving and the
  mismatch is visible in `/system/overview`.

---

## 7. Tests and CI (there were none)

* `tests/` — 5 suites, **97 tests**: auth, API contract,
  pipeline/schema/validation/parity, uncertainty, model persistence. The suite is
  hermetic: temporary user store, audit DB, signing key and observation store, so
  it leaves the repository untouched and is safe to run repeatedly.
* Every operational endpoint is asserted to return **401** anonymously and the
  admin endpoints **403** for a low-privilege token.
* `.github/workflows/ci.yml` — installs both requirement files, compiles
  everything, runs the suite, then **boots the API** and smoke-tests
  `/health`, `/login`, `/predict/manual`, and that `/graph` is closed.

```bash
pytest -q
```

---

## 8. Provenance is visible

`/system/overview` reports `provenance`, `provenance_label`, `persistence` and
`uncertainty`; the dashboard header shows a badge (**🧪 Simulated data**) that
switches automatically once real observations are ingested. Dependencies are
pinned in `requirements.txt` (dev tools in `requirements-dev.txt`), and secrets,
audit databases and ingested operator data are git-ignored.

---

## What Phase 1 deliberately does *not* claim

* The model still learns a **simulator**, not the railway. The infrastructure to
  fix that exists (ingest → validate → parity → provenance); the data does not,
  because no live NTES feed is openly available.
* Exceedance probabilities are interpolated from nine quantiles; they are
  honest about saturation but they are not a full predictive distribution.
* The demo account `admin`/`swr2026` still exists so the app runs offline, and the
  server warns about it on every boot until `SWR_USER`/`SWR_PASS` are set.
