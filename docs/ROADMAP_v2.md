# RailPulse v2 — Upgrade Plan: from impressive demo to real-world decision system

> Review date: 2026-10-08 · Reviewed against commit `a02f0e7` (`main`)
> Scope: what the app does **today**, where it diverges from how Indian Railways control
> rooms actually work, and a prioritised set of upgrades that close those gaps.

---

## 1. Executive summary

RailPulse today is a **well-engineered, genuinely novel research demo**: it treats the rail
network as a graph, flattens NetworkX centralities into tabular features, trains 10 classical
models, and proves via ablation that graph-derived features cut MAE by ~46–53%. The engineering
is real — a working FastAPI backend, a 2,500-line interactive dashboard, a rule-based Copilot,
Docker/Render packaging, an auto-retraining model bundle.

But measured against **the real-world problem** — helping a Section Controller decide what to do
in the next 30 minutes when a train is running late — there are three structural gaps:

| # | Gap | Why it matters in the real world |
|---|-----|----------------------------------|
| **G1** | **All training data is synthetic** (`src/data_generator.py` simulates a delay field with exponential distance decay). | The model has learned the *simulator's* physics, not railway physics. It cannot yet be validated against, or used on, real traffic. Every accuracy claim is a claim about simulated data. |
| **G2** | **The app predicts but does not prescribe.** A dispatcher cannot act on "48 min arrival delay". They need *"hold 2244 at Arsikere for 6 min — that saves 41 network delay-minutes and breaks no connections."* | Prediction without a ranked, costed action plan is not decision support. This is the single biggest value upgrade available. |
| **G3** | **No uncertainty.** Only point estimates (MAE/RMSE/R²). | Operational decisions are threshold decisions ("will it miss the 15-min connection?") and need *probabilities*, not point estimates. |

Plus a set of **credibility and safety defects** that would block any real deployment or a hard
evaluation review:

| Defect | Evidence in repo | Real-world consequence |
|---|---|---|
| API is unauthenticated | `src/app.py` — `grep -c "Depends\|verify_token\|require_auth"` → **0**. `/login` returns `{"success": true}` and the dashboard gates itself **client-side** (`dashboard.html` ~line 1759). | Every endpoint (`/predict/*`, `/export/*`, `/copilot/chat`) is open to anyone. Login is theatre. |
| Silent auto-retrain on import | `src/inference.py::ensure_bundle()` retrains data → features → models *at import time* if the version or pickle doesn't match. | In production a library bump would silently replace the live model mid-shift. No audit, no rollback. |
| Fragile model persistence | Server log on boot: XGBoost `If you are loading a serialized model ... please export the model by calling Booster.save_model from that version first`. | Models are version-coupled joblib pickles; a dependency change loses the model. |
| Zero tests | No `tests/`, no CI. | No regression safety net for any of the upgrades below. |
| Metrics are ML-only | `results/model_metrics.csv` has MAE/RMSE/R². | No operator KPI (punctuality %, delay-minutes/100 trains, crew-hours lost, ₹ cost of delay). Controllers speak in those units. |
| No crew / rake (rolling-stock) cascade | Data schema has `train_id`, `origin`, `destination` only — no rake link, no crew roster. | The **second** cascade axis is missing: a late incoming rake delays the *return* trip, and crew can run out of duty hours. This is a top real cause of knock-on delays. |
| No precedence / block-section / platform logic | Edges are `(a, b, km)` only. | Real cascade physics is headway, block-section occupancy, loop-line crossings, platform availability, precedence (who overtakes whom). Without it, "cascade" is distance decay, not operations. |
| No alerting, no audit trail, no drift monitoring | — | Nothing tells the control office; nothing records what was predicted vs. what happened. |

**Bottom line:** the *research spine* (graph → tabular → model → ablation) is strong and worth
keeping. What is missing is everything between "a model that scores well on synthetic data" and
"a system a controller can trust on a bad foggy Tuesday".

---

## 2. Upgrade packs

Each pack is self-contained and independently shippable. Effort is for one focused engineer.

### P1 — Real data ingestion & model honesty

**Problem.** The app cannot consume real railway data and does not disclose that it is
simulated. Both are fixable.

**Build.**
1. `src/sources/base.py` — a `RailDataSource` protocol emitting **canonical journey records**
   (train no., type, station code, scheduled/actual arrival+departure, platform, delay, cause head).
2. Adapters:
   - `CsvSource` / `ExcelSource` — dispatcher-exported logs, with a **column mapping UI** so any
     operator's format works (station name → code resolution against `config.STATIONS`).
   - `WebhookSource` — `POST /ingest/realtime` accepting live position/delay updates and folding
     them into the rolling congestion features.
   - `PublicDatasetSource` — loaders for open datasets (e.g. data.gov.in punctuality/traffic
     datasets). Note: official **NTES has no public API**; live feeds require an authorised
     partner or third-party tracker — the adapter interface is designed so either can drop in
     without touching the model layer.
3. `src/sources/validate.py` — a quality gate: unknown station codes, impossible speeds,
   duplicate rows, stale timestamps, completeness score. Reject-and-report rather than silently train.
4. **Delay-cause attribution.** Indian Railways records causes under lettered heads
   (pre-occupied line, rolling-stock failure, engineering/TRT, conventions, natural causes…).
   Ingesting them lets the app answer *"what is actually causing delay on this section?"* — which
   is how a real fix gets prioritised — instead of only *"how late will it be?"*.
5. **Provenance + parity.** Store `data_provenance` in the model bundle (`synthetic` |
   `synthetic+real` | `real`), surface it as a badge in the dashboard header, and add a
   **synthetic-vs-real distribution test** (KS statistic per feature) that fails loudly if the
   real data diverges from what the model was trained on.

**Impact:** high (removes the fatal credibility gap) · **Effort:** M–L (mostly schema plumbing;
real feeds need credentials you must supply)

---

### P2 — Uncertainty-aware forecasting

**Problem.** Point estimates cannot answer threshold questions.

**Build.**
1. Quantile models: LightGBM `objective="quantile"` at α = 0.1/0.5/0.9, plus the existing
   point ensemble as the median fallback.
2. **Conformal calibration** on a held-out split (split-conformal / CQR) so 80% and 90%
   intervals have *guaranteed* marginal coverage — a defensible statistical claim.
3. Operational outputs: `P(delay > 15 / 30 / 60 min)` → **connection-miss risk** and
   **knock-on risk**, shown as a band on the cascade card.
4. Evaluation: PICP (coverage), MPIW (sharpness), pinball loss, plus a reliability diagram in
   `results/figures/`. Calibration is measurable, so this is easy to defend.

**Impact:** high · **Effort:** M · **Files:** `src/train.py`, `src/inference.py`, `dashboard.html`

---

### P3 — From prediction to prescription: the mitigation optimiser  ⭐ highest real-world value

**Problem.** The Copilot lists mitigation *suggestions*; nothing computes the best action and
quantifies its benefit.

**Build.** `src/planner.py`:
- **Action space** (real dispatcher levers): hold train *t* for *k* minutes at station *s*;
  overtake via loop line; re-platform; divert via an alternate route (`nx.shortest_simple_paths`
  already available through `graph_utils`); short-terminate / turn back early; skip a stop.
- **Cost model** (configurable, in `config.py`): delay-minutes × train priority weight,
  connection-miss penalty, crew overtime (HOER duty limits), platform-reallocation risk, fuel.
- **Optimiser**: greedy + local search (or an OR-Tools MIP if you accept the dependency)
  minimising total *network* delay-minutes + cost penalty, subject to feasibility.
- **Output**: a ranked action plan — *"Hold 2244 at Arsikere 6 min → net −41 delay-minutes,
  0 connections broken, crew within duty hours"* — plus an explicit **no-action counterfactual**
  so the controller sees the delta, not just the plan.
- **API**: `POST /plan/mitigation` and a new "Action Plan" panel on the cascade card.

**Impact:** highest — this is what turns RailPulse from a predictor into a decision-support
system · **Effort:** L · **Depends on:** P4 for credible counterfactuals

---

### P4 — Digital twin: block-section forward simulation + Marey charts

**Problem.** Cascade logic is "distance decay", not railway operations. And there is no way to
see the next 30 minutes.

**Build.** `src/simulator.py`:
1. **Section model**: block sections per edge, headway, single/double line, loop-line length,
   platform count, max permissible speed per section. Extend `EDGES` to section records
   (backwards-compatible with the existing `(a, b, km)` tuples).
2. **Discrete-time simulator**: 15-second or 1-minute steps; trains move by block section;
   occupancy and headway constraints enforced; precedence resolved by priority class + input order.
3. **Calibration** from data: fit the delay-vs-headway saturation curve so simulated queues match
   observed congestion — the simulator must be *trained*, not hand-waved.
4. **Marey / string-line chart** (time on x, distance on y): the standard dispatcher visual, and
   a very strong addition to the UI. Show planned vs. simulated trajectories and available
   crossing loops.
5. Roll-forward 15/30/60/120 min from a live incident → projected network state; re-run with a
   P3 action plan applied → **quantified, causally-grounded benefit**.

**Impact:** high (unlocks P3's credibility and a killer visual) · **Effort:** L

---

### P5 — Train-level cascade: rake turnaround & crew duty hours

> **Partly delivered (2026-10-08).** The train-level cascade is live: a
> reconstructed timetable (`src/timetable.py`) plus an operating-rule conflict
> engine (`src/conflicts.py`) now answer *"how many and which upcoming trains does
> this delay affect?"* — `POST /cascade/trains`, the **Affected Trains** panel in
> the dashboard, and a Copilot tool. Delivered: `FOLLOWING_BLOCK`,
> `PLATFORM_REGULATION`, `RAKE_TURNAROUND` (the rake half of this pack), with a
> probability per train from the calibrated quantiles. Still open: **crew duty
> hours** (needs a crew-linkage model), and single-/double-line attributes that
> would turn `MEET_REGULATION` from an assumption into a finding.

**Problem.** Only the spatial cascade is modelled. In reality a delayed rake delays the *return
trip*, and a crew can exceed duty hours long before the train is late enough to notice.

**Build.** `src/rolling_stock.py`:
- **Rake linkage**: inbound train → next scheduled trip (turnaround minimum, primary maintenance
  at the home shed, sick-line risk) → propagate delay down the *train cycle*, not just the route.
- **Crew model**: roster, signing-on station, duty start, HOER duty-hour limit (~10 h on duty for
  running staff, configurable), and spare-crew availability at relief points. Output: *"crew of
  12627 will exceed duty hours in 2 h 40 m — needs relief at Birur"*.
- Feed both back as features (`rake_turnaround_slack`, `crew_hours_remaining`) — a genuinely new
  feature family, not a repackaging of centralities.

**Impact:** high (a whole missing cascade axis) · **Effort:** M–L

---

### P6 — Operational hardening & governance

**Build.**
1. **Real auth**: HMAC/JWT session tokens, hashed passwords (today: plaintext in `USERS`),
   roles (`dispatcher` / `controller` / `admin`), login rate-limiting, and a FastAPI dependency
   guarding every endpoint — replacing the client-side gate.
2. **Audit log**: append-only record of who queried what, what was predicted, and what action was
   taken (SQLite is enough). Essential for post-incident review and for regulated deployment.
3. **Model registry + drift monitor**: versioned bundles, champion/challenger, PSI/KS drift on
   live vs. training features, and a **deliberate, logged, reversible retrain** to replace
   `ensure_bundle()`'s silent behaviour.
4. **Tests + CI**: pytest suite (pipeline smoke, API contract, predictor determinism, planner
   feasibility) wired to GitHub Actions. Today there are **zero** tests.
5. **Alerting**: threshold webhooks (SMS/Slack/Telegram/email) when a projected cascade crosses a
   severity line, plus a daily ops digest.

**Impact:** high for credibility/deployability · **Effort:** M

---

### P7 — Scale, cost KPIs and control-room UX

**Build.**
1. **Operational KPI dashboard**: punctuality %, delay-minutes per 100 trains, **₹ cost of delay**
   (configurable per train-hour), crew-hours lost, cancellations avoided, trend vs. target, and a
   zone/junction league table. Replace ML metrics with operator metrics as the headline numbers.
2. **Scale beyond 84 stations**: build the graph from open station/route data for the full IR
   network (7,000+ stations, 13,000+ daily services). Requires incremental centrality caching and
   a parquet/SQLite store instead of a 30 MB joblib bundle.
3. **True geometry**: real rail alignments instead of straight lines between stations, so the map
   looks like a railway rather than a wireframe.
4. **Control-room UX**: keyboard-first console, dense information layout, tablet layout, and
   **Hindi/regional i18n** for field staff.

**Impact:** medium–high · **Effort:** L (scale item is XL)

---

## 3. Recommended sequencing

**Phase 1 — "Credit" (make it honest and safe):** P6.1/P6.4 first (auth + tests), then P1
provenance and real-data adapters, then P2 intervals.
*You can now say, truthfully: "trained on X, uncertain to ±Y, access-controlled, tested."*

**Phase 2 — "Prescription" (make it useful):** P4 simulator (minimal viable: headway + block
sections + Marey chart) → P3 optimiser → counterfactual savings.

**Phase 3 — "Scale & operations":** P5 rake/crew cascade, P6.2/6.3/6.5 (audit, registry, alerts),
P7 KPIs, then network scale-up.

### The demo that would land hardest

> A fog-affected evening peak. The controller asks for a projection; the twin rolls 60 minutes
> forward and shows three junctions turning red in a Marey chart. The optimiser proposes: *hold
> 12627 for 4 min at Arsikere, overtake 2244 at Birur loop* → **net −47 delay-minutes, ₹1.8 L
> avoided, zero connections broken, all crews within duty hours** — with an 85% conformal
> interval on the headline number and every step written to the audit log.

That single flow exercises P2+P3+P4+P5+P6 and is the whole "real-world problem" in one screen.

---

## 4. Honest caveats

- **Live NTES data is not openly available.** Real feeds need an authorised partner, a paid
  provider, or an operator's own exports. P1 is therefore built as an *adapter + quality gate*,
  not a scraper — so it works with whatever feed you legitimately obtain.
- **Cost coefficients (₹ per train-hour, HOER limits, headway values) are operator-specific.**
  They belong in a config file with defensible defaults, clearly documented as assumptions.
- **Simulator calibration needs incident logs.** Without historical data the digital twin is
  still useful for *relative* comparisons (Plan A vs Plan B) but not for absolute minute claims.
- **This sandbox cannot reach map-tile or railway-data hosts** (egress is limited to package
  registries). Everything above is buildable and testable here except the live-feed path, which
  needs your credentials and a network-enabled host.

---

## 5. Immediate small wins (< 1 hour each)

1. Guard the API with a real token dependency — the current login is decorative.
2. Pin dependency versions and export models natively (`save_model` / `Booster` JSON) to kill the
   XGBoost pickle-loading warnings.
3. Add a **"Trained on: synthetic data"** badge to the dashboard header. Honesty costs nothing and
   buys enormous credibility with any reviewer.
4. Make `ensure_bundle()`'s retrain explicit, logged and opt-in (`SWR_AUTO_RETRAIN=1`).
5. Add a `tests/` directory with a pipeline smoke test + `/health` contract test.
6. Un-hardcode the credentials: require `SWR_USER`/`SWR_PASS` (or a seeded hash) instead of the
   default `admin`/`swr2026` shipped in source.
