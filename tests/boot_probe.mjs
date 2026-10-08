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

const documentStub = {
  getElementById: () => element(),
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

// --- load the dashboard's inline script -------------------------------------
const html = readFileSync(process.argv[2] || "src/dashboard.html", "utf8");
const blocks = [...html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)];
if (!blocks.length) { console.error("no inline script found"); process.exit(1); }

const failures = [];
const finish = () => {
  const before = calls.filter(c => !c.path.startsWith("/static") && c.path !== "/login");
  const protectedBeforeSignIn = before.filter(c =>
    PROTECTED.has(c.path) || [...PROTECTED].some(p => c.path.startsWith(p + "/")));

  console.log(`requests during cold boot: ${calls.length}`);
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

try {
  // Execute the page script; top-level statements run immediately (as in a browser).
  new Function(blocks[0][1])();
  // The boot() IIFE is async — give its microtasks a turn to settle.
  setTimeout(finish, 400);
} catch (error) {
  console.error("script failed to execute:", error.message);
  process.exit(1);
}
