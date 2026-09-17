"""
Dispatcher Copilot — a natural-language agent over the live railway network.

Two operating modes (auto-selected):

  * ``offline`` (default)  – a deterministic tool-calling engine: it parses the
    dispatcher's message, picks the right tools (prediction, cascade ripple,
    choke-point ranking, routing, mitigations, what-if simulation), runs them
    against the live predictor/graph and composes a cited, readable answer.
    Needs nothing but the installed packages.

  * ``llm`` – when an OpenAI-compatible API key is present
    (``COPILOT_LLM_KEY`` / ``OPENAI_API_KEY``), the same tool set is exposed to
    the model via JSON tool-calling, and the model composes the final answer.
    Any failure falls back to the offline engine.

The tool layer is shared by both modes, so answers always come from the real
NetworkX graph + trained models — never hallucinated network facts.
"""
from __future__ import annotations

import json
import math
import os
import re

import requests

from graph_utils import (
    cascade_alert,
    path_distance_km,
    ripple_projection,
    shortest_path_km,
)

# ---------------------------------------------------------------------------
# Station aliases (common spellings dispatchers actually type)
# ---------------------------------------------------------------------------
ALIASES = {
    "bangalore": "KSR Bengaluru", "bengaluru": "KSR Bengaluru", "ksr": "KSR Bengaluru",
    "sbc": "KSR Bengaluru", "bengaluru city": "KSR Bengaluru",
    "cantonment": "Bengaluru Cantonment", "bengaluru cantonment": "Bengaluru Cantonment",
    "bnc": "Bengaluru Cantonment",
    "yesvantpur": "Yesvantpur", "yeshwanthpur": "Yesvantpur", "ypr": "Yesvantpur",
    "kr puram": "Krishnarajapuram", "krpuram": "Krishnarajapuram", "kjm": "Krishnarajapuram",
    "whitefield": "Whitefield", "yelahanka": "Yelahanka",
    "kengeri": "Kengeri", "bidadi": "Bidadi",
    "ramanagara": "Ramanagara", "ramanagaram": "Ramanagara",
    "channapatna": "Channapatna", "maddur": "Maddur", "mandya": "Mandya",
    "srirangapatna": "Srirangapatna", "srirangapattana": "Srirangapatna",
    "mysuru": "Mysuru", "mysore": "Mysuru", "mys": "Mysuru",
    "nanjangud": "Nanjangud", "chamarajanagar": "Chamarajanagar",
    "krishnarajanagara": "Krishnarajanagara",
    "hassan": "Hassan", "hole narsipur": "Hole Narsipur", "holenarsipur": "Hole Narsipur",
    "channarayapatna": "Channarayapatna",
    "sakleshpur": "Sakleshpur", "sakleshpura": "Sakleshpur",
    "subrahmanya": "Subrahmanya Road", "subrahmanya road": "Subrahmanya Road",
    "bantawala": "Bantawala", "bantwal": "Bantawala",
    "tumakuru": "Tumakuru", "tumkur": "Tumakuru", "tk": "Tumakuru",
    "gubbi": "Gubbi", "tiptur": "Tiptur", "arsikere": "Arsikere",
    "kadur": "Kadur", "birur": "Birur",
    "davangere": "Davangere", "harihar": "Harihar", "ranibennur": "Ranibennur",
    "haveri": "Haveri",
    "hubballi": "Hubballi", "hubli": "Hubballi", "ubl": "Hubballi",
    "dharwad": "Dharwad", "alnavar": "Alnavar", "londa": "Londa",
    "khanapur": "Khanapur",
    "belagavi": "Belagavi", "belgaum": "Belagavi", "bgm": "Belagavi",
    "gadag": "Gadag", "koppal": "Koppal",
    "hosapete": "Hosapete", "hospet": "Hosapete", "hosapet": "Hosapete",
    "toranagallu": "Toranagallu",
    "ballari": "Ballari", "bellary": "Ballari",
    "bangarapet": "Bangarapet", "bangarpet": "Bangarapet",
    "kolar": "Kolar", "malur": "Malur", "hosur": "Hosur",
    "chitradurga": "Chitradurga", "chikkajajur": "Chikkajajur",
    "dodballapur": "Dodballapur", "doddaballapur": "Dodballapur",
    "chikkaballapur": "Chikkaballapur", "channasandra": "Channasandra",
    "chennai": "Chennai Central", "madras": "Chennai Central", "mas": "Chennai Central",
    "renigunta": "Renigunta", "tirupati": "Tirupati",
    "vijayawada": "Vijayawada", "bza": "Vijayawada",
    "visakhapatnam": "Visakhapatnam", "vizag": "Visakhapatnam", "vskp": "Visakhapatnam",
    "secunderabad": "Secunderabad", "hyderabad": "Secunderabad", "sc": "Secunderabad",
    "warangal": "Warangal", "guntakal": "Guntakal", "raichur": "Raichur",
    "kalaburagi": "Kalaburagi", "gulbarga": "Kalaburagi", "bidar": "Bidar",
    "nagpur": "Nagpur", "bhopal": "Bhopal",
    "agra": "Agra Cantt", "agra cantt": "Agra Cantt",
    "nizamuddin": "Hazrat Nizamuddin", "delhi": "Hazrat Nizamuddin",
    "ndls": "Hazrat Nizamuddin", "nzm": "Hazrat Nizamuddin",
    "jaipur": "Jaipur", "ahmedabad": "Ahmedabad",
    "mumbai": "Mumbai CSMT", "cst": "Mumbai CSMT", "csmt": "Mumbai CSMT",
    "bombay": "Mumbai CSMT", "pune": "Pune", "solapur": "Solapur",
    "aurangabad": "Aurangabad",
    "mangaluru": "Mangaluru Central", "mangalore": "Mangaluru Central",
    "kozhikode": "Kozhikode", "calicut": "Kozhikode", "palakkad": "Palakkad",
    "ernakulam": "Ernakulam", "cochin": "Ernakulam", "kochi": "Ernakulam",
    "coimbatore": "Coimbatore", "cbe": "Coimbatore", "erode": "Erode", "salem": "Salem",
    "tiruchirappalli": "Tiruchirappalli", "trichy": "Tiruchirappalli",
    "tpj": "Tiruchirappalli", "madurai": "Madurai",
    "thiruvananthapuram": "Thiruvananthapuram", "trivandrum": "Thiruvananthapuram",
    "tvc": "Thiruvananthapuram",
}

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
WEATHERS = {"clear": "Clear", "rain": "Rain", "rainy": "Rain", "fog": "Fog",
            "foggy": "Fog", "storm": "Storm", "stormy": "Storm"}
TRAIN_TYPES = {"express": "Express", "superfast": "Superfast", "intercity": "Intercity",
               "passenger": "Passenger", "memu": "MEMU", "freight": "Freight",
               "shatabdi": "Superfast", "vande bharat": "Superfast", "vandebharat": "Superfast"}

# ---------------------------------------------------------------------------
# LLM configuration (optional)
# ---------------------------------------------------------------------------
def _llm_config() -> dict | None:
    key = os.environ.get("COPILOT_LLM_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    return {
        "key": key,
        "base": os.environ.get("COPILOT_LLM_BASE", "https://api.openai.com/v1").rstrip("/"),
        "model": os.environ.get("COPILOT_LLM_MODEL", "gpt-4o-mini"),
    }


TOOL_SCHEMA = """[
  {"name":"analyze", "description":"Full delay analysis for a station: predicted delay, cascade ripple and mitigations.",
   "args":{"current_station":"str","destination":"str?","train_type":"str?","hour":"int?","day":"str?","weather":"str?","current_delay_min":"int?"}},
  {"name":"choke", "description":"Rank junctions/stations most likely to choke first (by betweenness).","args":{"k":"int?"}},
  {"name":"route", "description":"Shortest rail route between two stations.","args":{"origin":"str","destination":"str"}},
  {"name":"station", "description":"Info about one station: degree, centralities, connected stations.","args":{"name":"str"}},
  {"name":"network", "description":"Network-wide status summary.","args":{}},
  {"name":"whatif", "description":"What-if: extra delay at a station; show impact delta.","args":{"station":"str","extra_minutes":"int","source":"str?","delay":"int?"}},
  {"name":"none", "description":"Greeting / general question that needs no tool.","args":{}}
]"""


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
class DispatcherCopilot:
    def __init__(self, predictor):
        self.p = predictor
        self.G = predictor.G
        self.cent = predictor.cent
        self.names = set(predictor.station_names())
        self.llm = _llm_config()

    # -- public API ---------------------------------------------------------
    def status(self) -> dict:
        if self.llm:
            return {"mode": "llm", "model": self.llm["model"],
                    "note": "LLM agent active — answers composed by the model from live tool results."}
        return {"mode": "offline", "model": "rule-engine",
                "note": "Offline agent active — deterministic tool-calling. Set COPILOT_LLM_KEY to enable an LLM."}

    def chat(self, message: str) -> dict:
        message = (message or "").strip()
        if not message:
            return {"reply": self._help(), "mode": self._mode(), "tools_used": []}

        if self.llm:
            try:
                return self._chat_llm(message)
            except Exception:  # noqa: BLE001 — fall back to the offline engine
                pass

        return self._chat_offline(message)

    # ------------------------------------------------------------------
    # Offline (deterministic) engine
    # ------------------------------------------------------------------
    def _mode(self) -> str:
        return "llm" if self.llm else "offline"

    def _chat_offline(self, message: str) -> dict:
        q = self._parse(message)
        stations = q["stations"]

        # greeting / help
        if not stations and not any(k in message.lower() for k in
                                    ("network", "status", "overview", "choke", "risk", "bottleneck")):
            return {"reply": self._help(), "mode": self._mode(), "tools_used": []}

        # what-if
        if re.search(r"what\s*[- ]?if|simulate|hold .* for", message.lower()):
            return self._answer_whatif(q, message)

        # two stations + route wording
        route_words = ("route", "path", "between", "distance", "from ", " to ", "via")
        if len(stations) >= 2 and any(w in message.lower() for w in route_words):
            return self._answer_route(stations[0], stations[1])

        # network overview / choke ranking
        if not stations and re.search(r"choke|bottleneck|riskiest|risk", message.lower()):
            return self._answer_choke(q.get("k", 5))

        if not stations and re.search(r"network|overview|status|summary", message.lower()):
            return self._answer_network()

        # single station info
        if len(stations) == 1 and re.search(r"info|about|tell me|details|degree|centrality|connections",
                                            message.lower()):
            return self._answer_station(stations[0])

        # default: full delay analysis (prediction + ripple + mitigations)
        if stations:
            return self._answer_analyze(stations, q, message)

        return {"reply": self._help(), "mode": self._mode(), "tools_used": []}

    # -- offline answers -----------------------------------------------------
    def _answer_analyze(self, stations, q, message) -> dict:
        src = stations[0]
        dest = stations[1] if len(stations) > 1 else q.get("destination")
        dest_explicit = dest is not None
        if dest == src:
            dest = None
            dest_explicit = False
        delay = q.get("delay", 0)
        hour = q.get("hour", 14)
        day = q.get("day", "Monday")
        weather = q.get("weather", "Clear")
        train = q.get("train_type", "Express")

        pred = self.p.predict(current_station=src, upcoming_station=None,
                              destination=dest, train_type=train, hour=hour,
                              day=day, weather=weather, current_delay_min=float(delay))
        alert = pred["alert"]
        level = alert["level"]
        top = alert["top_risk_station"]
        at_risk = alert["at_risk_stations"]
        ripples = ripple_projection(self.G, src, float(delay), radius=3)

        lines = [f"**{src}** reporting **{int(delay)} min** delay "
                 f"({day} {hour}:00, {weather}, {train})."]
        lines.append("")
        lines.append(f"Predicted destination arrival delay: **{pred['predicted_destination_arrival_delay_min']} min** "
                     f"(model: {pred['model']}).")
        if dest_explicit and pred["route"] and len(pred["route"]) > 1:
            lines.append(f"Route: {' → '.join(pred['route'])} ({pred['route_km']} km).")

        lines.append("")
        lines.append(f"Cascade alert: **{level}** ({alert['tag']}). "
                     f"{len(at_risk)} downstream stations projected above 15 min threshold.")

        if ripples:
            lines.append("")
            lines.append("**Junctions that choke first:**")
            for i, r in enumerate(ripples[:4], 1):
                lines.append(f"- {i}. **{r['station']}** — ~{r['projected_impact_min']} min at hop {r['hop']} "
                             f"(eigenvector {r['eigenvector_centrality']}).")

        m = self._mitigations(src, dest, delay, pred, level, top, ripples)
        lines.append("")
        lines.append("**Recommended mitigations:**")
        lines.extend(f"- {x}" for x in m)

        return {
            "reply": "\n".join(lines),
            "mode": self._mode(),
            "tools_used": ["predict", "ripple", "mitigate"],
            "map_action": {"type": "cascade", "source": src, "route": pred["route"] or [],
                           "delay": int(delay), "pred": pred["predicted_destination_arrival_delay_min"],
                           "level": level},
        }

    def _mitigations(self, src, dest, delay, pred, level, top, ripples) -> list[str]:
        out = []
        hop1 = [r for r in ripples if r["hop"] == 1][:3]
        if level in ("CRITICAL", "HIGH"):
            if hop1:
                names = ", ".join(r["station"] for r in hop1)
                out.append(f"Regulate departures at **{names}** — hold outbound trains "
                           f"{max(10, min(25, int(delay) // 2))} min to relieve the section.")
            if top:
                out.append(f"Priority clearing + staff at **{top}** (the highest-impact downstream junction).")
            alt = self._diversion(src, dest, top)
            if alt:
                out.append(f"Divert the affected train via **{' → '.join(alt)}** "
                           f"({path_distance_km(self.G, alt):.0f} km) to bypass the hotspot.")
            else:
                out.append("No shorter diversion exists — keep the planned route and absorb the delay.")
        else:
            out.append("No immediate intervention needed — continue monitoring downstream stations.")
            if hop1:
                out.append(f"Watch **{', '.join(r['station'] for r in hop1[:2])}** for knock-on delays.")
        if ripples:
            out.append(f"Expect ripple decay to ~{ripples[-1]['projected_impact_min']} min at the 3-hop edge.")
        return out

    def _diversion(self, src, dest, avoid) -> list[str] | None:
        if not dest or avoid == src or avoid == dest:
            return None
        import networkx as nx
        G2 = self.G.copy()
        G2.remove_node(avoid)
        try:
            alt = nx.shortest_path(G2, source=src, target=dest, weight="km")
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return None
        original = path_distance_km(self.G, shortest_path_km(self.G, src, dest) or [])
        alt_km = path_distance_km(self.G, alt)
        if original and alt_km <= original * 1.35:
            return alt
        return None

    def _answer_route(self, a, b) -> dict:
        path = shortest_path_km(self.G, a, b)
        if not path:
            return {"reply": f"Sorry — no connected route found between **{a}** and **{b}**.",
                    "mode": self._mode(), "tools_used": ["route"]}
        km = path_distance_km(self.G, path)
        legs = []
        for i in range(len(path) - 1):
            legs.append(f"{path[i]} → {path[i+1]} ({self.G[path[i]][path[i+1]]['km']:.0f} km)")
        reply = (f"**{a} → {b}** — {len(path)-1} stops, **{km:.0f} km**.\n\n"
                 + "\n".join(f"- {l}" for l in legs))
        return {"reply": reply, "mode": self._mode(), "tools_used": ["route"],
                "map_action": {"type": "route", "source": a, "route": path, "pred": 0, "delay": 0, "level": "LOW"}}

    def _answer_choke(self, k) -> dict:
        ranked = sorted(self.G.nodes(), key=lambda n: self.cent.get(n, {}).get("betweenness_centrality", 0), reverse=True)
        top = ranked[:max(1, min(k, 10))]
        reply = "**Junctions most likely to choke first** (highest betweenness centrality):\n\n"
        for i, n in enumerate(top, 1):
            c = self.cent.get(n, {})
            reply += (f"- {i}. **{n}** — betweenness {c.get('betweenness_centrality', 0):.4f}, "
                      f"degree {self.G.degree(n)}, eigenvector {c.get('eigenvector_centrality', 0):.4f}\n")
        return {"reply": reply.rstrip(), "mode": self._mode(), "tools_used": ["choke"]}

    def _answer_network(self) -> dict:
        n = self.G.number_of_nodes()
        e = self.G.number_of_edges()
        ranked = sorted(self.G.nodes(), key=lambda x: self.cent.get(x, {}).get("betweenness_centrality", 0), reverse=True)
        hubs = sorted(self.G.nodes(), key=lambda x: self.G.degree(x), reverse=True)
        reply = (f"**Network status** — {n} stations, {e} track segments.\n\n"
                 f"Top choke point: **{ranked[0]}** (betweenness {self.cent.get(ranked[0], {}).get('betweenness_centrality', 0):.4f}).\n"
                 f"Top hub: **{hubs[0]}** (degree {self.G.degree(hubs[0])}).\n"
                 f"Highest eigenvector influence: **{max(self.G.nodes(), key=lambda x: self.cent.get(x, {}).get('eigenvector_centrality', 0))}**.")
        return {"reply": reply, "mode": self._mode(), "tools_used": ["network"]}

    def _answer_station(self, name) -> dict:
        c = self.cent.get(name, {})
        nbrs = sorted(self.G.neighbors(name))
        reply = (f"**{name}** — degree **{self.G.degree(name)}**.\n\n"
                 f"Eigenvector centrality: {c.get('eigenvector_centrality', 0):.4f}\n"
                 f"Betweenness: {c.get('betweenness_centrality', 0):.4f}\n"
                 f"PageRank: {c.get('pagerank', 0):.4f}\n\n"
                 f"Connected to ({len(nbrs)}): {', '.join(nbrs[:12])}{'…' if len(nbrs) > 12 else ''}")
        return {"reply": reply, "mode": self._mode(), "tools_used": ["station"]}

    def _answer_whatif(self, q, message) -> dict:
        src = q["stations"][0] if q["stations"] else None
        hold = q["stations"][1] if len(q["stations"]) > 1 else src
        extra = q.get("extra_minutes", 0)
        base_delay = q.get("delay", 0)
        if not src:
            return {"reply": "For a what-if I need a station — e.g. “What if I hold Mysuru 15 extra minutes?”",
                    "mode": self._mode(), "tools_used": []}
        baseline = ripple_projection(self.G, src, float(base_delay), radius=3)
        if hold == src:
            scenario = ripple_projection(self.G, src, float(base_delay + extra), radius=3)
        else:
            scenario = baseline + ripple_projection(self.G, hold, float(extra), radius=3)
        merged = {}
        for r in scenario:
            merged[r["station"]] = max(merged.get(r["station"], 0), r["projected_impact_min"])
        order = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
        alert = cascade_alert(self.G, src, float(base_delay) + float(extra))
        reply = (f"**What-if:** add **{int(extra)} min** at **{hold}** "
                 f"(combined with {int(base_delay)} min at {src}).\n\n"
                 f"New alert level: **{alert['level']}** ({alert['tag']}).\n\n"
                 f"Hardest-hit stations after the change:\n")
        for i, (s, v) in enumerate(order[:4], 1):
            b = next((r["projected_impact_min"] for r in baseline if r["station"] == s), 0)
            d = v - b
            reply += f"- {i}. **{s}** — ~{v:.1f} min ({'+' if d >= 0 else ''}{d:.1f} vs baseline)\n"
        return {"reply": reply.rstrip(), "mode": self._mode(), "tools_used": ["whatif"]}

    # ------------------------------------------------------------------
    # LLM engine (tool-calling)
    # ------------------------------------------------------------------
    def _chat_llm(self, message: str) -> dict:
        tool_call = self._llm_pick_tool(message)
        tool = tool_call.get("tool", "none")
        args = tool_call.get("args", {}) or {}

        if tool == "none":
            return {"reply": tool_call.get("answer") or self._help(), "mode": "llm", "tools_used": []}

        result = self._run_tool(tool, args)
        reply = self._llm_compose(message, result["summary"])
        return {"reply": reply, "mode": "llm", "tools_used": [tool], **result.get("extra", {})}

    def _llm_pick_tool(self, message: str) -> dict:
        sys = ("You are the RailPulse Dispatcher Copilot. Choose ONE tool from the schema and return ONLY JSON "
               f"of the form {{'tool': <name>, 'args': {{...}}}} or {{'tool':'none','answer':'...'}} for greetings.\n"
               f"Schema: {TOOL_SCHEMA}")
        raw = self._llm_complete(sys, message, temperature=0)
        js = self._extract_json(raw)
        if not js or "tool" not in js:
            return {"tool": "none"}
        return {"tool": js.get("tool"), "args": js.get("args", {})}

    def _llm_compose(self, message: str, summary: str) -> str:
        sys = ("You are the RailPulse Dispatcher Copilot. Using ONLY the tool result below, write a concise, "
               "actionable answer for the dispatcher. Cite the numbers. Use short paragraphs and bullet lists. "
               "Do not invent facts absent from the tool result.")
        user = f"Question: {message}\n\nTool result:\n{summary}"
        return self._llm_complete(sys, user, temperature=0.3).strip() or summary

    def _llm_complete(self, system: str, user: str, temperature: float) -> str:
        r = requests.post(
            f"{self.llm['base']}/chat/completions",
            headers={"Authorization": f"Bearer {self.llm['key']}", "Content-Type": "application/json"},
            json={"model": self.llm["model"], "temperature": temperature,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]},
            timeout=25,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        text = re.sub(r"```(?:json)?", "", text).strip()
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    # -- shared tool layer ---------------------------------------------------
    def _run_tool(self, tool: str, args: dict) -> dict:
        if tool == "analyze":
            q = self._parse(" ".join(str(v) for v in args.values()))
            stations = q["stations"] or [args.get("current_station")]
            src = args.get("current_station") or (stations[0] if stations else None)
            if not src:
                return {"summary": "No station identified.", "extra": {}}
            src = self._resolve(src)
            dest = self._resolve(args.get("destination")) if args.get("destination") else None
            pred = self.p.predict(current_station=src, upcoming_station=None, destination=dest,
                                  train_type=args.get("train_type", "Express"), hour=int(args.get("hour", 14)),
                                  day=args.get("day", "Monday"), weather=args.get("weather", "Clear"),
                                  current_delay_min=float(args.get("current_delay_min", 0)))
            summary = json.dumps({"alert": pred["alert"], "predicted_arrival_delay_min": pred["predicted_destination_arrival_delay_min"],
                                  "route": pred["route"], "route_km": pred["route_km"], "model": pred["model"]})
            extra = {"map_action": {"type": "cascade", "source": src, "route": pred["route"] or [],
                                    "delay": int(args.get("current_delay_min", 0)),
                                    "pred": pred["predicted_destination_arrival_delay_min"], "level": pred["alert"]["level"]}}
            return {"summary": summary, "extra": extra}
        if tool == "choke":
            k = int(args.get("k", 5))
            ranked = sorted(self.G.nodes(), key=lambda n: self.cent.get(n, {}).get("betweenness_centrality", 0), reverse=True)[:k]
            summary = json.dumps([{"station": n, "betweenness": self.cent.get(n, {}).get("betweenness_centrality", 0),
                                   "degree": self.G.degree(n)} for n in ranked])
            return {"summary": summary, "extra": {}}
        if tool == "route":
            a, b = self._resolve(args.get("origin")), self._resolve(args.get("destination"))
            path = shortest_path_km(self.G, a, b)
            summary = json.dumps({"path": path, "distance_km": path_distance_km(self.G, path) if path else None})
            extra = {"map_action": {"type": "route", "source": a, "route": path or [], "pred": 0, "delay": 0, "level": "LOW"}} if path else {}
            return {"summary": summary, "extra": extra}
        if tool == "station":
            name = self._resolve(args.get("name"))
            c = self.cent.get(name, {})
            summary = json.dumps({"station": name, "degree": self.G.degree(name), "neighbours": list(self.G.neighbors(name)),
                                  "eigenvector": c.get("eigenvector_centrality", 0),
                                  "betweenness": c.get("betweenness_centrality", 0), "pagerank": c.get("pagerank", 0)})
            return {"summary": summary, "extra": {}}
        if tool == "network":
            summary = json.dumps({"stations": self.G.number_of_nodes(), "edges": self.G.number_of_edges()})
            return {"summary": summary, "extra": {}}
        if tool == "whatif":
            station = self._resolve(args.get("station"))
            src = self._resolve(args.get("source")) or station
            extra = int(args.get("extra_minutes", 0))
            base = int(args.get("delay", 0))
            ripples = ripple_projection(self.G, station, float(extra), radius=3)
            alert = cascade_alert(self.G, src, float(base + extra))
            summary = json.dumps({"extra_at": station, "extra_minutes": extra, "new_level": alert["level"],
                                  "top_impacts": ripples[:4]})
            return {"summary": summary, "extra": {}}
        return {"summary": "Unknown tool.", "extra": {}}

    # ------------------------------------------------------------------
    # NLU helpers
    # ------------------------------------------------------------------
    def _parse(self, message: str) -> dict:
        low = message.lower()
        stations = self._find_stations(low)
        delay = self._extract_delay(low)
        hour = self._extract_hour(low)
        day = next((d.title() for d in DAYS if d in low), "Monday")
        weather = next((v for k, v in WEATHERS.items() if k in low), "Clear")
        train = next((v for k, v in TRAIN_TYPES.items() if k in low), "Express")
        k = 5
        mk = re.search(r"top\s*(\d+)", low)
        if mk:
            k = int(mk.group(1))
        extra = 0
        me = re.search(r"(\d{2,3})\s*(?:extra|more)", low)
        if me:
            extra = int(me.group(1))
        return {"stations": stations, "delay": delay, "hour": hour, "day": day,
                "weather": weather, "train_type": train, "k": k, "extra_minutes": extra}

    def _find_stations(self, low: str) -> list[str]:
        norm = re.sub(r"[^a-z0-9 ]", " ", low)
        found = []  # (pos, canonical)
        # 1) aliases
        for alias, canonical in ALIASES.items():
            pos = norm.find(alias)
            if pos != -1 and canonical in self.names:
                found.append((pos, canonical))
        # 2) exact-ish substring of full names
        for name in self.names:
            key = name.lower()
            pos = norm.find(key)
            if pos != -1:
                found.append((pos, name))
        if not found:
            return []
        # sort by position, dedupe preserving order
        found.sort(key=lambda t: t[0])
        seen, out = set(), []
        for _pos, name in found:
            if name not in seen:
                seen.add(name)
                out.append(name)
        return out

    def _resolve(self, name: str) -> str:
        if not name:
            return name
        if name in self.names:
            return name
        low = name.lower()
        if low in ALIASES and ALIASES[low] in self.names:
            return ALIASES[low]
        matches = self._find_stations(low)
        return matches[0] if matches else name

    @staticmethod
    def _extract_delay(low: str) -> int:
        m = re.search(r"(\d{2,3})\s*(?:min|mins|minutes)", low)
        if m:
            return int(m.group(1))
        m = re.search(r"delay(?:ed)?\s*(?:by)?\s*(\d{2,3})", low)
        if m:
            return int(m.group(1))
        m = re.search(r"(\d{2,3})\s*min", low)
        if m:
            return int(m.group(1))
        if "delay" in low:
            return 0
        return 0

    @staticmethod
    def _extract_hour(low: str) -> int:
        m = re.search(r"\b(\d{1,2}):(\d{2})\b", low)
        if m:
            return int(m.group(1))
        m = re.search(r"\b(\d{1,2})\s*(?:am|pm)\b", low)
        if m:
            h = int(m.group(1)) % 12
            return h + 12 if "pm" in low else h
        m = re.search(r"(?:at|around|by)\s+(\d{1,2})\b", low)
        if m:
            return int(m.group(1))
        return 14

    def _help(self) -> str:
        return ("I'm the **RailPulse Dispatcher Copilot**. Ask me things like:\n\n"
                "- “Mysuru delayed 45 min at 18:00 — which junctions choke first?”\n"
                "- “What's my best mitigation for a cascade at KSR Bengaluru?”\n"
                "- “Route from Chennai Central to Hubballi”\n"
                "- “Top 5 riskiest stations right now”\n"
                "- “Tell me about Guntakal”\n"
                "- “What if I hold Mysuru 20 extra minutes?”\n\n"
                "I run the live prediction model, the NetworkX cascade engine and routing over the real graph, "
                "then answer with the numbers.")
