/**
 * Cold-boot probe: does the dashboard call any protected endpoint before sign-in?
 *
 * A browser is the only place the dashboard really runs, so this loads the inline
 * script from dashboard.html into Node with a stub DOM, records every fetch, and
 * replays the *server's* authorisation rules against it. It then simulates a
 * successful sign-in and checks the app actually loads afterwards.
 *
 * Regression target: `loadCopilotStatus()` used to run at page-load time and probe
 * a dispatcher-only route, so every cold load raised a 401 and showed a
 * "session expired" alarm before the user could type a password.
 *
 * Usage:  node tests/boot_probe.mjs src/dashboard.html
 * Exit code 0 = pass, 1 = fail.
 */
import { readFileSync } from "node:fs";

// --- routes that require a session (mirrors src/app.py role gates) -----------
const PROTECTED = new Set([
  "/graph", "/stations", "/models", "/exports", "/system/overview", "/history",
  "/model/metrics", "/demo/records", "/copilot/status", "/copilot/chat",
  "/ingest/status", "/data/causes", "/data/parity", "/admin/audit", "/admin/users",
  "/auth/me", "/auth/logout", "/predict/manual", "/predict/upload", "/predict/batch",
  "/gps/nearest", "/gps/route", "/gps/predict", "/model/select",
]);
const PUBLIC = new Set(["/health", "/auth/config", "/login", "/register", "/"]);

// --- stub DOM ---------------------------------------------------------------
const calls = [];
let signedIn = false;

function element() {
  const target = { style: {}, classList: { add() {}, remove() {}, toggle() {} },
                   dataset: {}, children: [], value: "", textContent: "", innerHTML: "" };
  return new Proxy(target, {
    get(t, prop) {
      if (prop in t) return t[prop];
      if (prop === "addEventListener" || prop === "removeEventListener") return () => {};
      if (prop === "appendChild" || prop === "remove" || prop === "focus") return () => {};
      if (prop === "querySelector" || prop === "querySelectorAll") return () => [];
      if (prop === "getContext") return () => null;
      if (prop === "getBoundingClientRect") return () => ({ top: 0, left: 0, width: 0, height: 0 });
      if (prop === "setAttribute" || prop === "getAttribute" || prop === "removeAttribute"
          || prop === "hasAttribute") return () => {};
      if (prop === "contains") return () => false;
      if (prop === "closest") return () => null;
      if (prop === "matches") return () => false;
      if (prop === Symbol.toPrimitive || prop === "toString") return () => "";
      if (prop === "then") return undefined;            // never look like a promise
      return t[prop] !== undefined ? t[prop] : undefined;
    },
    set(t, prop, value) { t[prop] = value; return true; },
  });
}

const ELEMENTS = new Map();
const byId = (id) => {
  if (!ELEMENTS.has(id)) ELEMENTS.set(id, Object.assign(element(), { id }));
  return ELEMENTS.get(id);
};

const documentStub = {
  getElementById: byId,
  querySelector: () => element(),
  querySelectorAll: () => [],
  createElement: () => element(),
  addEventListener: () => {},
  body: element(),
  head: element(),
  documentElement: element(),
  createTextNode: () => element(),
  getElementsByTagName: () => [],
  getElementsByClassName: () => [],
};

// A fetch that applies the server's rules: protected + no session => 401.
globalThis.fetch = async (url, opts = {}) => {
  const path = String(url).split("?")[0];
  const auth = (opts.headers && (opts.headers.Authorization || opts.headers.authorization)) || "";
  const hasCredential = signedIn || /^Bearer \S+/.test(auth);
  const protectedRoute = PROTECTED.has(path)
    || [...PROTECTED].some(p => path.startsWith(p + "/"));
  calls.push({ path, credential: hasCredential });

  if (protectedRoute && !hasCredential) {
    return {
      status: 401, ok: false,
      headers: { get: () => "application/json" },
      clone() { return this; },
      json: async () => ({ detail: "No session credential arrived with this request." }),
      text: async () => "",
    };
  }
  const body = path === "/login"
    ? { success: true, token: "v1.test", role: "admin", full_name: "Dispatcher" }
    : path === "/cascade/trains" ? AFFECTED_FIXTURE
    : path === "/auth/config" ? { enabled: true, default_demo_account_active: true }
    : path.endsWith(".csv") ? "a,b\n1,2" : {};
  return {
    status: 200, ok: true,
    headers: { get: () => (path.endsWith(".csv") ? "text/csv" : "application/json") },
    clone() { return this; },
    json: async () => body,
    text: async () => (typeof body === "string" ? body : JSON.stringify(body)),
  };
};

const storage = () => ({ getItem: () => null, setItem() {}, removeItem() {}, clear() {} });
globalThis.window = { addEventListener: () => {}, matchMedia: () => ({ matches: false, addEventListener() {} }),
                      location: { href: "http://localhost/", origin: "http://localhost" }, speechSynthesis: undefined };
globalThis.document = documentStub;
globalThis.localStorage = storage();
globalThis.sessionStorage = storage();
// `navigator` is a getter-only global in Node 22 — define the fields instead.
Object.defineProperty(globalThis, "navigator", {
  value: { userAgent: "node", geolocation: undefined, mediaDevices: undefined },
  configurable: true, writable: true,
});
globalThis.location = globalThis.window.location;
globalThis.addEventListener = () => {};
globalThis.requestAnimationFrame = (fn) => setTimeout(fn, 0);
globalThis.cancelAnimationFrame = () => {};
globalThis.getComputedStyle = () => ({ getPropertyValue: () => "" });
globalThis.alert = () => {};
globalThis.L = new Proxy({}, { get: () => () => element() });   // Leaflet stub
globalThis.SpeechRecognition = undefined;
globalThis.webkitSpeechRecognition = undefined;
globalThis.URL = URL;
globalThis.Blob = class { constructor() {} };
globalThis.FileReader = class { readAsText() {} };
globalThis.speechSynthesis = undefined;

const AFFECTED_FIXTURE = {
  delayed_train: { at: "Mysuru", destination: "KSR Bengaluru",
                   current_delay_min: 45, predicted_destination_delay_min: 78 },
  affected_count: 2,
  by_severity: { severe: 1, moderate: 1, minor: 0 },
  by_kind: { FOLLOWING_BLOCK: 2 },
  assumed_conflicts: 0,
  horizon: { minutes: 120, analysed_from: "18:40", analysed_until: "20:45",
             day_name: "Monday", candidates_considered: 40, trains_in_timetable: 10000 },
  rule_set: { headway_min: 3, platform_window_min: 5, turnaround_buffer_min: 25,
              recovery_factor: 0.6 },
  assumptions: { caveat: "rebuildable test fixture", probability_note: "test" },
  affected_trains: [
    { train_id: "TRN07926", train_type: "Superfast", kind: "FOLLOWING_BLOCK",
      where: "Maddur -> Channapatna", when: "19:50", expected_added_delay_min: 14.2,
      p_over_15min: 0.52, severity: "severe", reason: "trails the delayed train",
      assumed: false },
    { train_id: "TRN03356", train_type: "Superfast", kind: "FOLLOWING_BLOCK",
      where: "Channapatna -> Ramanagara", when: "20:25", expected_added_delay_min: 13.0,
      p_over_15min: 0.32, severity: "moderate", reason: "holds for the block",
      assumed: false },
  ],
};

// --- load the dashboard's inline script -------------------------------------
const html = readFileSync(process.argv[2] || "src/dashboard.html", "utf8");
const blocks = [...html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)];
if (!blocks.length) { console.error("no inline script found"); process.exit(1); }

const failures = [];
const finish = (bootCalls) => {
  // Only the cold-boot requests matter for the ordering rule; the panel check
  // deliberately issues an authenticated call *after* boot.
  const before = bootCalls.filter(c => !c.path.startsWith("/static") && c.path !== "/login");
  const protectedBeforeSignIn = before.filter(c =>
    PROTECTED.has(c.path) || [...PROTECTED].some(p => c.path.startsWith(p + "/")));

  console.log(`requests during cold boot: ${bootCalls.length}`);
  for (const c of calls) console.log(`   ${c.path}${c.credential ? "  [credential]" : ""}`);

  if (protectedBeforeSignIn.length) {
    failures.push(`${protectedBeforeSignIn.length} protected endpoint(s) called before sign-in: `
      + protectedBeforeSignIn.map(c => c.path).join(", "));
  } else {
    console.log("PASS: no protected endpoint was called before sign-in");
  }
  const publicOnly = before.every(c => PUBLIC.has(c.path) || c.path.startsWith("/static")
                                    || c.path.startsWith("/lib") || c.path.startsWith("/tiles"));
  if (!publicOnly) failures.push("a non-public endpoint was called without a credential");

  if (failures.length) { failures.forEach(f => console.error("FAIL: " + f)); process.exit(1); }
  console.log("PASS: cold boot issues no authenticated request");
};

async function panelChecks() {
  const api = globalThis.__railpulse;
  if (!api) { failures.push("the page did not expose its internals to the probe"); return; }

  // Signed in, then ask for the affected-trains panel exactly as predict() does.
  api.AUTH.set("v1.test");
  signedIn = true;
  await api.loadAffectedTrains({ current_station: "Mysuru", destination: "KSR Bengaluru",
                                 train_type: "Superfast", hour: 18, day: "Monday",
                                 weather: "Clear", current_delay_min: 45 });
  const html = byId("affectedTrains").innerHTML || "";
  if (!html.includes("AFFECTED TRAINS")) failures.push("panel did not render a heading");
  if (!html.includes("2 in the next 120 min")) failures.push("panel did not show the count");
  if (!html.includes("TRN07926") || !html.includes("TRN03356"))
    failures.push("panel did not list the affected trains");
  if (!html.includes("+14.2 min")) failures.push("panel did not show the expected delay");
  if (!html.includes("52%")) failures.push("panel did not show the exceedance probability");
  if (!html.includes("18:40")) failures.push("panel did not show the analysis window");
  if (!html.includes("Operating assumptions"))
    failures.push("panel did not disclose the assumptions it rests on");
  if (html.includes("undefined")) failures.push("panel rendered 'undefined' somewhere");
  if (!failures.some(f => f.startsWith("panel")))
    console.log(`PASS: affected-trains panel rendered (${html.length} chars of markup)`);
}

try {
  // Execute the page script; top-level statements run immediately (as in a browser).
  // Expose the internals at the end so the probe can drive them like a user would.
  new Function(blocks[0][1] +
    "\n;globalThis.__railpulse = { loadAffectedTrains, renderAffectedTrains, AUTH };")();
  // The boot() IIFE is async — give its microtasks a turn, then drive the panel.
  setTimeout(async () => {
    const bootCalls = calls.slice();      // what the page did on its own
    await panelChecks();                  // then what the panel does when asked
    finish(bootCalls);
  }, 400);
} catch (error) {
  console.error("script failed to execute:", error.message);
  process.exit(1);
}
