"""
Global configuration for the Graph-Derived Feature Boosting project
(Cascading Train Delay Prediction in the South Western Railway zone).

Everything here is shared by the data generator, feature engineering,
training, ablations, and the FastAPI / Gradio apps.
"""
from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Authentication (dispatcher login). Override via environment variables:
#   SWR_USER / SWR_PASS
# ----------------------------------------------------------------------------
AUTH_USER = os.environ.get("SWR_USER", "admin")
AUTH_PASS = os.environ.get("SWR_PASS", "swr2026")

# ----------------------------------------------------------------------------
# CORS allow-list (was "*"). Same-origin clients (the bundled dashboard) always
# work; add hosts here for separately deployed clients.
# ----------------------------------------------------------------------------
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get(
        "SWR_ALLOWED_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000,http://localhost:7860,"
        "http://127.0.0.1:7860",
    ).split(",") if o.strip()
]

# Self-registration creates a low-privilege (dispatcher) account. Fine for a
# demo/classroom deployment; switch it off for anything real.
ALLOW_SELF_REGISTRATION = os.environ.get("SWR_ALLOW_SELF_REGISTRATION", "1") == "1"

# Bump this whenever the network topology changes: the app will detect a
# version mismatch and retrain automatically so stale graphs are never served.
NETWORK_VERSION = "2-national"

# ----------------------------------------------------------------------------
# Paths (all relative to the project root so the app runs from anywhere)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
EXPORTS_DIR = ROOT / "exports"

for _d in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, RESULTS_DIR, FIGURES_DIR, EXPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_CSV = RAW_DIR / "swr_journeys.csv"
FEATURES_CSV = PROCESSED_DIR / "features.csv"
BUNDLE_PATH = MODEL_DIR / "bundle.joblib"          # graph + models + metadata
METRICS_CSV = RESULTS_DIR / "model_metrics.csv"
ABLATION_CSV = RESULTS_DIR / "ablation_results.csv"
IMPORTANCE_CSV = RESULTS_DIR / "feature_importance.csv"

# ----------------------------------------------------------------------------
# Categorical encodings (fixed, shared between training and inference)
# ----------------------------------------------------------------------------
TRAIN_TYPES = ["Express", "Superfast", "Intercity", "Passenger", "MEMU", "Freight"]
WEATHERS = ["Clear", "Rain", "Fog", "Storm"]

TYPE_ENC = {t: i for i, t in enumerate(TRAIN_TYPES)}
WEATHER_ENC = {w: i for i, w in enumerate(WEATHERS)}

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_ENC = {d: i for i, d in enumerate(DAYS)}

# Feature groups (used by ablations and the inference pipeline)
ISOLATED_FEATURES = ["train_type_enc", "scheduled_hour", "day_of_week", "weather_enc"]
STATIC_GRAPH_FEATURES = [
    "edge_weight_next_km",
    "current_station_degree",
    "current_station_degree_centrality",
    "current_station_eigenvector_centrality",
    "current_station_pagerank",
    "current_station_betweenness_centrality",
    "current_station_closeness_centrality",
    "upcoming_station_degree",
    "upcoming_station_eigenvector_centrality",
    "upcoming_station_pagerank",
]
DYNAMIC_NETWORK_FEATURES = ["congestion_r1", "congestion_r2", "congestion_r3", "cascading_delay_index"]
CLUSTER_FEATURES = ["delay_cluster"]

FULL_FEATURES = (
    ISOLATED_FEATURES + STATIC_GRAPH_FEATURES + DYNAMIC_NETWORK_FEATURES + CLUSTER_FEATURES
)
TARGET = "destination_arrival_delay_min"

# ----------------------------------------------------------------------------
# Data provenance
#
# The published model is trained on a SIMULATED delay field
# (``data_generator.py``) because no live NTES feed is openly available. That is
# a legitimate modelling choice for a research prototype — but it is not
# acceptable for the app to present simulated performance as if it were
# measured on real traffic. This value travels into the model bundle, the API
# and the dashboard badge, and ``sources/ingest.py`` upgrades it automatically
# when genuine data is loaded.
#
#   "synthetic"       - trained only on the simulator
#   "synthetic+real"  - simulator used for pre-training, real data for fine-tuning
#   "real"            - trained only on observed railway data
# ----------------------------------------------------------------------------
PROVENANCE = os.environ.get("SWR_PROVENANCE", "synthetic")
PROVENANCE_LABELS = {
    "synthetic": "Simulated dataset (no live NTES feed available)",
    "synthetic+real": "Mixed — simulator for pre-training, real logs for fine-tuning",
    "real": "Observed railway data",
}

# Conformal uncertainty artefacts
UNCERTAINTY_CSV = RESULTS_DIR / "uncertainty_metrics.csv"
CALIBRATION_JSON = MODEL_DIR / "calibration.json"
# Delay -> congestion relationship fitted on the training sample. Written by
# train.py, read by both inference (when no live feed supplies congestion) and
# ingestion (when a real feed omits it), so both sides speak the same feature
# space the models were fitted on.
CONGESTION_PROXY_JSON = MODEL_DIR / "congestion_proxy.json"
STATION_CLUSTERS_JSON = PROCESSED_DIR / "station_clusters.json"  # DBSCAN labels per station
ENSEMBLE_DIR = MODEL_DIR / "ensemble"        # fused-ensemble components (portable formats)

# Auto-retraining is now explicit: a version mismatch is reported and does NOT
# silently replace the served model unless this is switched on.
AUTO_RETRAIN = os.environ.get("SWR_AUTO_RETRAIN", "0") == "1"

# ----------------------------------------------------------------------------
# Weather / train-type delay penalties (used by the data generator)
# ----------------------------------------------------------------------------
WEATHER_PENALTY = {"Clear": 0, "Rain": 8, "Fog": 12, "Storm": 25}
TYPE_PENALTY = {
    "Express": 2, "Superfast": 1, "Intercity": 3, "Passenger": 10, "MEMU": 8, "Freight": 15,
}


def peak_load(hour: int) -> float:
    """Relative network load for a given hour (rush-hour amplification)."""
    if 7 <= hour <= 10 or 17 <= hour <= 20:
        return 1.6
    if 11 <= hour <= 16:
        return 1.2
    if 21 <= hour <= 23 or 0 <= hour <= 5:
        return 0.6
    return 1.0


def peak_penalty(hour: int) -> float:
    if 7 <= hour <= 10 or 17 <= hour <= 20:
        return 8.0
    if 11 <= hour <= 16 or hour == 21:
        return 3.0
    return 0.0


# ----------------------------------------------------------------------------
# The South Western Railway sub-grid: ~53 real stations (node) + track (edge).
# Edge weight = rail distance in km. Coordinates are approximate (for DBSCAN /
# heatmap visualisation only).
# ----------------------------------------------------------------------------
STATIONS: dict[str, tuple[float, float]] = {
    # Bengaluru cluster
    "KSR Bengaluru":          (12.9782, 77.5946),
    "Bengaluru Cantonment":   (12.9909, 77.6038),
    "Yesvantpur":             (13.0225, 77.5507),
    "Krishnarajapuram":       (12.9989, 77.6759),
    "Whitefield":             (12.9699, 77.7495),
    "Channasandra":           (13.0015, 77.6450),
    "Yelahanka":              (13.1006, 77.5964),
    "Kengeri":                (12.9180, 77.4829),
    "Bidadi":                 (12.7970, 77.3947),
    "Dodballapur":            (13.2876, 77.5536),
    "Chikkaballapur":         (13.4355, 77.7273),
    # Mysuru line
    "Ramanagara":             (12.7150, 77.2905),
    "Channapatna":            (12.6508, 77.2070),
    "Maddur":                 (12.5820, 77.0430),
    "Mandya":                 (12.5221, 76.8970),
    "Srirangapatna":          (12.4215, 76.6929),
    "Mysuru":                 (12.3144, 76.6450),
    "Nanjangud":              (12.1170, 76.6839),
    "Chamarajanagar":         (11.9241, 76.9397),
    "Krishnarajanagara":      (12.4634, 76.3645),
    # Hassan / Mangaluru line
    "Hassan":                 (13.0072, 76.0964),
    "Hole Narsipur":          (12.7864, 76.2385),
    "Channarayapatna":        (12.9009, 76.3896),
    "Sakleshpur":             (12.9412, 75.7850),
    "Subrahmanya Road":       (12.6799, 75.5258),
    "Bantawala":              (12.8966, 75.0319),
    # Hubballi trunk
    "Tumakuru":               (13.3372, 77.1012),
    "Gubbi":                  (13.3124, 76.9416),
    "Tiptur":                 (13.2560, 76.4775),
    "Arsikere":               (13.3136, 76.2585),
    "Kadur":                  (13.5475, 76.0121),
    "Birur":                  (13.5980, 75.9721),
    "Davangere":              (14.4644, 75.9212),
    "Harihar":                (14.5124, 75.8063),
    "Ranibennur":             (14.6181, 75.6202),
    "Haveri":                 (14.7936, 75.4034),
    "Hubballi":               (15.3647, 75.1239),
    "Dharwad":                (15.4361, 74.9940),
    "Alnavar":                (15.4272, 74.7415),
    "Londa":                  (15.4696, 74.5317),
    "Khanapur":               (15.6399, 74.5139),
    "Belagavi":               (15.8497, 74.4977),
    # Gadag - Ballari branch
    "Gadag":                  (15.4260, 75.6392),
    "Koppal":                 (15.3470, 76.1566),
    "Hosapete":               (15.2685, 76.3882),
    "Toranagallu":            (15.2050, 76.7310),
    "Ballari":                (15.1394, 76.9240),
    # Bengaluru - Kolar line
    "Bangarapet":             (12.9986, 78.1743),
    "Kolar":                  (13.1362, 78.1329),
    "Malur":                  (13.0043, 77.9355),
    # Border / branch
    "Hosur":                  (12.7414, 77.8226),
    "Chitradurga":            (14.2221, 76.3950),
    "Chikkajajur":            (14.2240, 76.1210),
    # ---------------- National trunk (Indian Railway Network) ---------------
    # South / East
    "Chennai Central":        (13.0827, 80.2707),
    "Renigunta":              (13.6325, 79.5122),
    "Tirupati":               (13.6288, 79.4192),
    "Vijayawada":             (16.5062, 80.6480),
    "Visakhapatnam":          (17.6868, 83.2185),
    "Secunderabad":           (17.4399, 78.4983),
    "Warangal":               (17.9784, 79.6000),
    "Guntakal":               (15.1664, 77.3754),
    "Raichur":                (16.2120, 77.3430),
    "Kalaburagi":             (17.3297, 76.8343),
    "Bidar":                  (17.9104, 77.5199),
    # Central / North
    "Nagpur":                 (21.1458, 79.0882),
    "Bhopal":                 (23.2599, 77.4126),
    "Agra Cantt":             (27.1767, 78.0081),
    "Hazrat Nizamuddin":      (28.5884, 77.2560),
    "Jaipur":                 (26.9124, 75.7873),
    "Ahmedabad":              (23.0225, 72.5714),
    # West
    "Mumbai CSMT":            (18.9402, 72.8355),
    "Pune":                   (18.5204, 73.8567),
    "Solapur":                (17.6599, 75.9064),
    "Aurangabad":             (19.8762, 75.3433),
    # Kerala / Tamil Nadu
    "Mangaluru Central":      (12.9141, 74.8560),
    "Kozhikode":              (11.2588, 75.7804),
    "Palakkad":               (10.7867, 76.6548),
    "Ernakulam":              (9.9312, 76.2673),
    "Coimbatore":             (11.0168, 76.9558),
    "Erode":                  (11.3410, 77.7172),
    "Salem":                  (11.6643, 78.1460),
    "Tiruchirappalli":        (10.7905, 78.7047),
    "Madurai":                (9.9252, 78.1198),
    "Thiruvananthapuram":     (8.5241, 76.9366),
}

# (station_a, station_b, distance_km)
EDGES: list[tuple[str, str, float]] = [
    # Bengaluru - Mysuru trunk
    ("KSR Bengaluru", "Kengeri", 12), ("Kengeri", "Bidadi", 17),
    ("Bidadi", "Ramanagara", 18), ("Ramanagara", "Channapatna", 30),
    ("Channapatna", "Maddur", 28), ("Maddur", "Mandya", 20),
    ("Mandya", "Srirangapatna", 18), ("Srirangapatna", "Mysuru", 15),
    ("Mysuru", "Nanjangud", 23), ("Nanjangud", "Chamarajanagar", 35),
    # Bengaluru - Hubballi trunk
    ("KSR Bengaluru", "Yesvantpur", 6), ("Yesvantpur", "Tumakuru", 64),
    ("Tumakuru", "Gubbi", 18), ("Gubbi", "Tiptur", 40), ("Tiptur", "Arsikere", 23),
    ("Arsikere", "Kadur", 37), ("Kadur", "Birur", 7), ("Birur", "Davangere", 103),
    ("Davangere", "Harihar", 14), ("Harihar", "Ranibennur", 20),
    ("Ranibennur", "Haveri", 31), ("Haveri", "Hubballi", 72),
    # Bengaluru - Kolar line
    ("KSR Bengaluru", "Krishnarajapuram", 15), ("Krishnarajapuram", "Channasandra", 3),
    ("Krishnarajapuram", "Whitefield", 12), ("Whitefield", "Malur", 27),
    ("Malur", "Bangarapet", 27), ("Bangarapet", "Kolar", 27),
    # Bengaluru - Hosur / Chikkaballapur
    ("Bengaluru Cantonment", "KSR Bengaluru", 4), ("Bengaluru Cantonment", "Hosur", 58),
    ("Bengaluru Cantonment", "Yelahanka", 16), ("Yelahanka", "Dodballapur", 18),
    ("Dodballapur", "Chikkaballapur", 26),
    # Hassan branch
    ("Arsikere", "Hassan", 47), ("Hassan", "Sakleshpur", 42),
    ("Sakleshpur", "Subrahmanya Road", 55), ("Subrahmanya Road", "Bantawala", 50),
    ("Hassan", "Channarayapatna", 37), ("Channarayapatna", "Hole Narsipur", 35),
    ("Hole Narsipur", "Krishnarajanagara", 30), ("Krishnarajanagara", "Mysuru", 34),
    # Hubballi - Belagavi
    ("Hubballi", "Dharwad", 20), ("Dharwad", "Alnavar", 34), ("Alnavar", "Londa", 32),
    ("Londa", "Khanapur", 26), ("Khanapur", "Belagavi", 27),
    # Hubballi - Gadag - Ballari
    ("Hubballi", "Gadag", 58), ("Gadag", "Koppal", 57), ("Koppal", "Hosapete", 28),
    ("Hosapete", "Toranagallu", 33), ("Toranagallu", "Ballari", 25),
    # Davangere - Chitradurga
    ("Davangere", "Chikkajajur", 50), ("Chikkajajur", "Chitradurga", 34),
    # ---------------- National trunk connections ---------------
    # South / East
    ("Bangarapet", "Chennai Central", 290),
    ("Chennai Central", "Renigunta", 135), ("Renigunta", "Tirupati", 10),
    ("Renigunta", "Guntakal", 230), ("Guntakal", "Raichur", 120),
    ("Raichur", "Kalaburagi", 95), ("Kalaburagi", "Solapur", 115),
    ("Guntakal", "KSR Bengaluru", 300),
    ("Secunderabad", "Warangal", 140), ("Warangal", "Vijayawada", 210),
    ("Vijayawada", "Visakhapatnam", 350), ("Vijayawada", "Chennai Central", 430),
    ("Secunderabad", "Guntakal", 300), ("Nagpur", "Secunderabad", 580),
    ("Kalaburagi", "Bidar", 110), ("Bidar", "Secunderabad", 120),
    # Central / North
    ("Nagpur", "Bhopal", 390), ("Bhopal", "Agra Cantt", 560),
    ("Agra Cantt", "Hazrat Nizamuddin", 195),
    ("Hazrat Nizamuddin", "Jaipur", 300), ("Jaipur", "Ahmedabad", 660),
    ("Ahmedabad", "Mumbai CSMT", 520),
    # West
    ("Solapur", "Pune", 250), ("Pune", "Mumbai CSMT", 190),
    ("Solapur", "Aurangabad", 200), ("Aurangabad", "Mumbai CSMT", 340),
    # Kerala / Tamil Nadu
    ("Bantawala", "Mangaluru Central", 20),
    ("Mangaluru Central", "Kozhikode", 230), ("Kozhikode", "Palakkad", 130),
    ("Palakkad", "Coimbatore", 55), ("Coimbatore", "Erode", 100),
    ("Erode", "Salem", 60), ("Salem", "Chennai Central", 330),
    ("Erode", "Tiruchirappalli", 140), ("Tiruchirappalli", "Madurai", 160),
    ("Madurai", "Thiruvananthapuram", 300),
    ("Palakkad", "Ernakulam", 150), ("Ernakulam", "Thiruvananthapuram", 220),
    ("Chamarajanagar", "Coimbatore", 90),
]

# Known high-traffic junctions whose delay propensity is amplified (delay sinks).
DELAY_SINK_BOOST = {
    "KSR Bengaluru": 1.6, "Bengaluru Cantonment": 1.4, "Yesvantpur": 1.4,
    "Arsikere": 1.3, "Hubballi": 1.2, "Mysuru": 1.1,
}
