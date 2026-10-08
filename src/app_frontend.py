"""
Gradio "Railway Dispatcher Dashboard" — Deep-Tech Noir aesthetic.

Access flow:
  1. LOGIN PAGE (username / password validated by the FastAPI backend).
  2. DISPATCHER DASHBOARD with four tabs:
       🎛 Manual Entry   - single live-status query -> cascade alert
       🗂 Dataset        - CSV upload -> batch predictions + export link
       🔗 Export Link    - download previously generated exports
       📊 Network & Model - grid map, ripple heatmap, metrics, ablations, SHAP

The dashboard talks to the FastAPI backend (default http://127.0.0.1:8000).
Run both together:  python src/run.py
"""
from __future__ import annotations

import os
import sys
import time

import gradio as gr
import pandas as pd
import requests

from config import (ABLATION_CSV, FIGURES_DIR, IMPORTANCE_CSV, METRICS_CSV)

BACKEND = os.environ.get("SWR_BACKEND_URL", "http://127.0.0.1:8000")

# ---------------------------------------------------------------------------
# Noir CSS
# ---------------------------------------------------------------------------
CSS = """
:root { --neon:#00e5ff; --accent:#ff2d95; --warn:#ffb300; --ok:#39ff88; }
.gradio-container {
    background:
        radial-gradient(circle at 20% 10%, rgba(0,229,255,0.08), transparent 40%),
        radial-gradient(circle at 80% 90%, rgba(255,45,149,0.08), transparent 40%),
        #05080f !important;
    color: #d7e2ee !important;
    font-family: 'Segoe UI', system-ui, sans-serif;
}
.gradio-container::before {
    content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
    background-image:
        linear-gradient(rgba(0,229,255,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,229,255,0.05) 1px, transparent 1px);
    background-size: 42px 42px;
}
footer {display:none !important;}
h1, h2, h3, label, .label-wrap span {color:#d7e2ee !important;}

/* ---- login card ---- */
#login-screen {
    max-width: 440px; margin: 48px auto; padding: 36px 40px 30px;
    border: 1px solid rgba(0,229,255,0.4); border-radius: 14px;
    background: rgba(7,12,22,0.9); box-shadow: 0 0 44px rgba(0,229,255,0.20);
    position: relative; z-index: 1;
}
#login-screen input {
    background: #0a1220 !important; color:#e8f4ff !important;
    border: 1px solid #1c2b3d !important; letter-spacing:.5px;
}
#login-screen input:focus { border-color: var(--neon) !important; }
#login-screen label span { color:#7f93a8 !important; }

button.primary, #predict_btn, #login_btn {
    background: linear-gradient(90deg, #00e5ff, #2979ff) !important;
    color: #001018 !important; font-weight:700 !important;
    border: none !important; letter-spacing:1px;
}
button.primary:hover, #predict_btn:hover, #login_btn:hover { filter:brightness(1.15); }

.alert-panel { border:1px solid var(--neon); border-radius:10px; padding:16px 18px;
    background: rgba(0,20,30,0.55); box-shadow: 0 0 24px rgba(0,229,255,0.18); }
.alert-panel.critical { border-color:#ff2d95; box-shadow:0 0 28px rgba(255,45,149,0.35); background:rgba(40,0,20,0.55); }
.alert-panel.high { border-color:#ffb300; box-shadow:0 0 24px rgba(255,179,0,0.3); }
.alert-panel.ok { border-color:#39ff88; box-shadow:0 0 20px rgba(57,255,136,0.25); }
.statbox { border:1px solid #1c2b3d; border-radius:10px; padding:12px 16px; background:rgba(10,16,26,0.7); }
.neon-title { color:var(--neon); text-shadow:0 0 12px rgba(0,229,255,0.55); }
.access-denied { color:#ff2d95; text-shadow:0 0 10px rgba(255,45,149,0.5); }
"""


def _alert_html(alert: dict, pred: int, model: str) -> str:
    level = alert.get("level", "LOW")
    tag = alert.get("tag", "NOMINAL")
    src = alert.get("source", "?")
    delay = alert.get("source_delay_min", 0)
    top = alert.get("top_risk_station", "—")
    risk = alert.get("at_risk_stations", [])
    rows = "".join(
        f"<li>{r['station']} — ~{r['projected_impact_min']} min (hop {r['hop']})</li>"
        for r in risk[:5]
    ) or "<li>No downstream stations projected above threshold.</li>"

    color = {"CRITICAL": "#ff2d95", "HIGH": "#ffb300",
             "MEDIUM": "#00e5ff", "LOW": "#39ff88"}.get(level, "#00e5ff")
    return f"""
    <div class="alert-panel {level.lower()}">
      <h3 style="color:{color};margin-top:0;">⚠ GRID CASCADE ALERT — {level} ({tag})</h3>
      <p><b>{delay:.0f}-min delay reported at {src}.</b><br>
      Predicted <b>destination arrival delay: {pred} min</b> (model: {model}).</p>
      <p>High probability of <b>20+ minute knock-on delays</b> converging at
      <span style="color:{color};font-weight:700;">{top}</span> within the next 90 minutes.</p>
      <p style="margin-bottom:4px;">Projected ripple (top stations):</p>
      <ul style="margin-top:2px;font-size:0.9em;">{rows}</ul>
    </div>
    """


# ---------------------------------------------------------------------------
# Authenticated backend client
#
# The API now requires a signed session token for every operational endpoint, so
# the console authenticates once with the configured service account and reuses
# that token. A 401 (expired token) triggers exactly one silent re-login before
# the error surfaces, which keeps a shift-long console session alive without
# asking the operator to sign in again.
# ---------------------------------------------------------------------------
_TOKEN: str = ""        # service account token (console → API)
_USER_TOKEN: str = ""  # the signed-in operator's token, preferred when present


def _login() -> str:
    """Obtain a service token from the backend (cached)."""
    global _TOKEN
    from config import AUTH_PASS, AUTH_USER
    try:
        r = requests.post(f"{BACKEND}/login",
                          json={"username": AUTH_USER, "password": AUTH_PASS},
                          timeout=8)
        if r.status_code == 200 and r.json().get("success"):
            _TOKEN = r.json().get("token", "")
    except Exception:  # noqa: BLE001
        _TOKEN = ""
    return _TOKEN


def _headers() -> dict:
    """Prefer the signed-in operator's token; fall back to the service account."""
    token = _USER_TOKEN or _TOKEN or _login()
    return {"Authorization": f"Bearer {token}"} if token else {}


def api_get(path: str, **kwargs):
    """GET with the service token, retrying once after re-authentication."""
    r = requests.get(f"{BACKEND}{path}", headers=_headers(), **kwargs)
    if r.status_code == 401:
        _login()
        r = requests.get(f"{BACKEND}{path}", headers=_headers(), **kwargs)
    return r


def api_post(path: str, **kwargs):
    r = requests.post(f"{BACKEND}{path}", headers=_headers(), **kwargs)
    if r.status_code == 401:
        _login()
        r = requests.post(f"{BACKEND}{path}", headers=_headers(), **kwargs)
    return r


def backend_up() -> bool:
    try:
        r = requests.get(f"{BACKEND}/health", timeout=3)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def _stations() -> list[str]:
    try:
        return api_get("/stations", timeout=5).json()["stations"]
    except Exception:  # noqa: BLE001
        return []


def _status_html() -> str:
    try:
        o = api_get("/system/overview", timeout=5).json()
        return f"""
        <div style="display:flex;gap:22px;flex-wrap:wrap;">
          <div class="statbox"><div style="color:#7f93a8;font-size:.7em;">ACTIVE MODEL</div>
            <b class="neon-title">{o['model']}</b></div>
          <div class="statbox"><div style="color:#7f93a8;font-size:.7em;">BEST SINGLE</div>
            <b>{o['best_single_model']}</b> · MAE {o['best_mae']} min</div>
          <div class="statbox"><div style="color:#7f93a8;font-size:.7em;">NETWORK</div>
            <b>{o['n_stations']} stations · {o['n_edges']} edges</b></div>
          <div class="statbox"><div style="color:#7f93a8;font-size:.7em;">GRAPH FEATURES</div>
            <b>{o['n_features']} flattened columns</b></div>
          <div class="statbox"><div style="color:#7f93a8;font-size:.7em;">TRAIN / TEST</div>
            <b>{o['n_train']} / {o['n_test']} journeys</b></div>
        </div>"""
    except Exception:  # noqa: BLE001
        return ""


def _err_html(msg: str) -> str:
    return f'<div class="access-denied" style="text-align:center;">⚠ {msg}</div>'


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------
def do_login(username: str, password: str):
    global _USER_TOKEN
    if not backend_up():
        return (_err_html("Backend offline — start FastAPI first (python src/run.py)."),
                gr.Column(visible=True), gr.Column(visible=False), "")
    try:
        r = requests.post(f"{BACKEND}/login",
                          json={"username": username, "password": password}, timeout=5)
        payload = r.json() if r.status_code == 200 else {}
        ok = bool(payload.get("success"))
    except Exception:  # noqa: BLE001
        ok, payload = False, {}
    if ok:
        # Use the operator's own token, so the API audit trail names the human
        # who asked, not the console's service account.
        _USER_TOKEN = payload.get("token", "")
        return "", gr.Column(visible=False), gr.Column(visible=True), _status_html()
    return (_err_html("ACCESS DENIED — invalid credentials"),
            gr.Column(visible=True), gr.Column(visible=False), "")


def do_logout():
    global _USER_TOKEN
    _USER_TOKEN = ""
    return gr.Column(visible=True), gr.Column(visible=False), ""


# ---------------------------------------------------------------------------
# Tab 1: manual entry
# ---------------------------------------------------------------------------
def predict_manual(current_station, upcoming_station, destination, train_type,
                   hour, day, weather, current_delay_min):
    if not backend_up():
        return (_alert_html({"level": "CRITICAL", "tag": "SEVERE", "source": current_station,
                             "source_delay_min": current_delay_min, "at_risk_stations": [],
                             "top_risk_station": "—"}, -1, "backend offline"),
                "Backend offline — start FastAPI first (python src/run.py).",
                "", None, None, "")
    payload = {
        "current_station": current_station, "upcoming_station": upcoming_station,
        "destination": destination, "train_type": train_type, "hour": int(hour),
        "day": day, "weather": weather, "current_delay_min": float(current_delay_min),
    }
    try:
        r = api_post("/predict/manual", json=payload, timeout=30)
        r.raise_for_status()
        res = r.json()
    except Exception as e:  # noqa: BLE001
        return (_alert_html({"level": "CRITICAL", "tag": "SEVERE", "source": current_station,
                             "source_delay_min": current_delay_min, "at_risk_stations": [],
                             "top_risk_station": "—"}, -1, "error"),
                f"Request failed: {e}", "", None, None, "")

    pred = res["predicted_destination_arrival_delay_min"]
    html = _alert_html(res["alert"], pred, res["model"])
    route = " → ".join(res.get("route") or []) or f"{current_station}"
    route_text = f"Route ({res.get('route_km', 0)} km): {route}"
    feats = res.get("features_used", {})

    # station intelligence panel
    try:
        meta = api_get(f"/stations/{current_station}", timeout=5).json()
        sinfo = (
            f"<div class='statbox'>"
            f"<b class='neon-title'>{meta['station']}</b><br>"
            f"Degree: <b>{meta['degree']}</b> · "
            f"Eigenvector centrality: <b>{meta['eigenvector_centrality']}</b> · "
            f"PageRank: <b>{meta['pagerank']}</b>"
            f"</div>"
        )
    except Exception:  # noqa: BLE001
        sinfo = ""
    return html, route_text, res["model"], res["alert"], feats, sinfo


# ---------------------------------------------------------------------------
# Tab 2: dataset upload
# ---------------------------------------------------------------------------
def process_dataset(file):
    if file is None:
        return None, "", "", "Upload a CSV first."
    if not backend_up():
        return None, "", "", "Backend offline — start FastAPI first."
    try:
        with open(file.name, "rb") as f:
            r = api_post("/predict/upload",
                         files={"file": ("input.csv", f, "text/csv")}, timeout=180)
        r.raise_for_status()
        data = r.json()
    except Exception as e:  # noqa: BLE001
        return None, "", "", f"Request failed: {e}"

    df = pd.DataFrame(data["rows"])
    if df.empty:
        return None, "", "", "No valid rows found in the uploaded CSV."

    export_id = data.get("export_id", "")
    link = f"{BACKEND}/export/{export_id}"
    table = df[["current_station", "destination", "predicted_arrival_delay_min",
                "alert_level", "top_risk_station"]]
    from config import EXPORTS_DIR
    local_path = EXPORTS_DIR / export_id
    return (table, link, str(local_path),
            f"Processed {len(df)} rows. Export: {export_id}")


# ---------------------------------------------------------------------------
# Tab 3: export links
# ---------------------------------------------------------------------------
def list_exports():
    from config import EXPORTS_DIR
    files = sorted(EXPORTS_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    rows = []
    for p in files:
        rows.append({
            "file": p.name,
            "rows": _count_lines(p),
            "backend_link": f"{BACKEND}/export/{p.name}",
        })
    return rows, [str(p) for p in files]


def _count_lines(p) -> int:
    try:
        with open(p) as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:  # noqa: BLE001
        return 0


# ---------------------------------------------------------------------------
# Tab 4: network & model insights (static files from results/)
# ---------------------------------------------------------------------------
def _img(path: str):
    return path if os.path.exists(path) else None


def _df(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------
THEME = gr.themes.Base(primary_hue="cyan", neutral_hue="slate")


def build_demo() -> gr.Blocks:
    stations = _stations()
    if not stations:
        stations = ["Mysuru", "KSR Bengaluru", "Kengeri", "Arsikere", "Hubballi"]

    # ---- static insights loaded from results/ at build time ----
    metrics_df = _df(METRICS_CSV)
    ablation_df = _df(ABLATION_CSV)
    importance_df = _df(IMPORTANCE_CSV).head(15)
    fig_dir = FIGURES_DIR

    with gr.Blocks(title="RailPulse — Dispatcher Dashboard") as demo:

        # ======================= LOGIN SCREEN =======================
        with gr.Column(visible=True, elem_id="login-screen") as login_screen:
            gr.HTML("""
            <div style="text-align:center;margin-bottom:8px;">
              <h1 class="neon-title" style="margin-bottom:2px;">◤ RAILPULSE DISPATCH CONTROL</h1>
              <p style="color:#7f93a8;margin-top:0;">Cascading Delay Prediction · South Western Railway zone</p>
              <div style="border-top:1px dashed #1c2b3d;margin:14px 0;"></div>
              <p style="color:#39ff88;letter-spacing:2px;font-size:.8em;">■ RESTRICTED — AUTHORIZED PERSONNEL ONLY ■</p>
            </div>""")
            username = gr.Textbox(label="Username", value="", elem_id="login_user")
            password = gr.Textbox(label="Password", type="password", elem_id="login_pw")
            login_btn = gr.Button("▶ ACCESS CONTROL", variant="primary", elem_id="login_btn")
            login_err = gr.HTML("")

        # ===================== MAIN DASHBOARD ======================
        with gr.Column(visible=False) as main_screen:
            with gr.Row():
                gr.HTML("""
                <div style="text-align:left;padding:2px 0;">
                  <h1 class="neon-title" style="margin-bottom:2px;">◤ RAILPULSE DISPATCHER DASHBOARD</h1>
                  <p style="color:#7f93a8;margin-top:0;">Graph-Derived Feature Boosting ·
                  Network-Aware Delay Prediction</p>
                </div>""")
                with gr.Column(scale=0, min_width=120):
                    logout_btn = gr.Button("⏻ LOGOUT")
            status_header = gr.HTML("")

            with gr.Tabs():
                # ---------------- Tab 1: Manual -------------------------------
                with gr.Tab("🎛 Manual Entry"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            current = gr.Dropdown(stations, label="Current Station",
                                                  value="Mysuru", allow_custom_value=False)
                            upcoming = gr.Dropdown([""] + stations, label="Upcoming Station (optional)",
                                                   value="", allow_custom_value=False)
                            destination = gr.Dropdown([""] + stations, label="Destination (optional)",
                                                      value="KSR Bengaluru", allow_custom_value=False)
                        with gr.Column(scale=1):
                            train_type = gr.Dropdown(["Express", "Superfast", "Intercity",
                                                      "Passenger", "MEMU", "Freight"],
                                                     label="Train Type", value="Express")
                            weather = gr.Dropdown(["Clear", "Rain", "Fog", "Storm"],
                                                  label="Weather at Current Station", value="Clear")
                            day = gr.Dropdown(["Monday", "Tuesday", "Wednesday", "Thursday",
                                               "Friday", "Saturday", "Sunday"],
                                              label="Day of Week", value="Monday")
                        with gr.Column(scale=1):
                            hour = gr.Slider(0, 23, value=14, step=1,
                                             label="Scheduled Hour (0–23)")
                            delay = gr.Slider(0, 120, value=45, step=5,
                                              label="Current Delay (minutes)")
                            run_btn = gr.Button("⚡ Run Cascade Prediction", variant="primary",
                                                elem_id="predict_btn")
                    alert = gr.HTML()
                    with gr.Row():
                        route_text = gr.Textbox(label="Route", interactive=False)
                        model_out = gr.Textbox(label="Model Used", interactive=False)
                    station_info = gr.HTML()
                    with gr.Accordion("Feature Snapshot (graph-flattened columns)", open=False):
                        feats_out = gr.JSON(label="Features fed to the regressor")
                    run_btn.click(predict_manual,
                                  [current, upcoming, destination, train_type, hour, day,
                                   weather, delay],
                                  [alert, route_text, model_out, feats_out, station_info])

                # ---------------- Tab 2: Dataset ------------------------------
                with gr.Tab("🗂 Dataset"):
                    gr.Markdown("""
                    **Upload a CSV of live train statuses.** Expected columns
                    (case-sensitive): `current_station, train_type, hour, day, weather,
                    current_delay_min, destination, upcoming_station`.
                    Only `current_station` is mandatory — the rest get sensible defaults.
                    """)
                    up = gr.File(label="Upload CSV", file_types=[".csv"])
                    proc_btn = gr.Button("⚡ Process Dataset", variant="primary")
                    with gr.Row():
                        out_table = gr.Dataframe(label="Batch Predictions", interactive=False)
                    with gr.Row():
                        link_out = gr.Textbox(label="Export Link (backend)", interactive=False)
                        status_out = gr.Textbox(label="Status", interactive=False)
                    dl = gr.File(label="Download Exported CSV")
                    proc_btn.click(process_dataset, [up],
                                   [out_table, link_out, dl, status_out])

                # ---------------- Tab 3: Export links -------------------------
                with gr.Tab("🔗 Export Link"):
                    gr.Markdown("Previously generated exports (most recent first).")
                    refresh_btn = gr.Button("↻ Refresh Exports")
                    exports_table = gr.Dataframe(
                        headers=["file", "rows", "backend_link"],
                        label="Export History", interactive=False)
                    exports_files = gr.File(label="Download Files", file_count="multiple")
                    refresh_btn.click(list_exports, [], [exports_table, exports_files])

                # ---------------- Tab 4: Network & Model ----------------------
                with gr.Tab("📊 Network & Model"):
                    gr.Markdown("### The graph-to-tabular pipeline, inspected")
                    with gr.Row():
                        gr.Image(_img(str(fig_dir / "01_network_map.png")),
                                 label="SWR grid — hub topology (eigenvector centrality)")
                        gr.Image(_img(str(fig_dir / "02_ripple_heatmap.png")),
                                 label="The cascade — ripple heatmap")
                    with gr.Row():
                        gr.Image(_img(str(fig_dir / "03_model_comparison.png")),
                                 label="10-model bake-off")
                        gr.Image(_img(str(fig_dir / "04_radius_learning_curve.png")),
                                 label="Radius of influence — decay of delay")
                    gr.Image(_img(str(fig_dir / "05_feature_importance.png")),
                             label="SHAP-style feature importance")
                    with gr.Row():
                        gr.Dataframe(metrics_df, label="Model Metrics (MAE / RMSE / R²)",
                                     interactive=False)
                        gr.Dataframe(ablation_df, label="Ablation Studies",
                                     interactive=False)
                    gr.Dataframe(importance_df,
                                 label="Feature Importance (MAE increase on shuffle)",
                                 interactive=False)
                    gr.Markdown("""
                    > **How to read this tab.** The network map shows the 53-station SWR grid
                    sized/coloured by eigenvector centrality (KSR Bengaluru dominates). The
                    ripple heatmap tells the story of a cascade: a delay at Mysuru turns its
                    neighbours red over successive hops. The bake-off table ranks the 10 models;
                    the radius curve proves the delay influence *decays* with distance
                    (the 3-edge ring adds only ~59% of the 2-edge ring's information); and the
                    importance chart shows congestion features outweigh weather — upending the
                    traditional isolated-delay assumptions.
                    """)

        # ---------------- events ----------------
        login_btn.click(do_login, [username, password],
                        [login_err, login_screen, main_screen, status_header])
        logout_btn.click(do_logout, [], [login_screen, main_screen, status_header])

    return demo


def launch(demo: gr.Blocks | None = None, share: bool | None = None) -> None:
    """Launch with the noir theme + CSS (Gradio 6 wants these on launch()).

    share=True creates a PUBLIC link via Gradio's tunnel (https://xxxx.gradio.live)
    so anyone on the internet can open the dashboard - no port-forwarding needed.

    NOTE: we intentionally do NOT enable demo.queue(). The preview proxy
    (and some corporate networks) drop the long-lived SSE/WebSocket stream the
    queue relies on, which makes Gradio show "Connection to the server was
    lost. Attempting to reconnect." Running without a queue executes each
    prediction synchronously over a plain HTTP request - far more robust
    behind proxies and perfectly fine for a dispatcher demo.
    """
    if demo is None:
        demo = build_demo()
    if share is None:
        share = os.environ.get("SWR_SHARE", "0") == "1" or "--share" in sys.argv
    demo.launch(
        server_name="0.0.0.0", server_port=7860, show_error=True,
        css=CSS, theme=THEME,
        ssr_mode=False,          # client-side render: avoids Gradio 6 SSR issues in iframes
        share=share,
    )


if __name__ == "__main__":
    launch()
