// Overlay + battle tracking. Collects each battle room's protocol lines and the
// |request| JSON, asks the local bridge for advice, renders the panel, and (in
// auto mode) sends the /choose command back through the page's socket.

const rooms = {}; // roomid -> { lines: [], request: null }
const settings = { mode: "deep", auto: false };
let sockjsMode = false;  // the official client wraps frames in SockJS framing
let seenTraffic = false;

chrome.storage.local.get(["psaMode", "psaAuto"], (v) => {
  if (v.psaMode) settings.mode = v.psaMode;
  if (typeof v.psaAuto === "boolean") settings.auto = v.psaAuto;
  renderControls();
});

document.addEventListener("psa-recv", (e) => {
  try {
    if (!seenTraffic) {
      seenTraffic = true;
      setStatus("connected — waiting for a battle…");
    }
    for (const msg of unwrapFrames(String(e.detail))) handleMessage(msg);
  } catch (err) { /* never break the client */ }
});

// The sim server speaks either raw protocol text or SockJS framing, depending
// on how the client connected: 'o' = open, 'h' = heartbeat, 'c[...]' = close,
// 'a["msg", ...]' = an array of protocol messages (JSON-escaped).
function unwrapFrames(data) {
  if (data === "o" || data.startsWith("h")) { sockjsMode = true; return []; }
  if (data.startsWith("c")) return [];
  if (data.startsWith("a")) {
    sockjsMode = true;
    try { return JSON.parse(data.slice(1)); } catch { return []; }
  }
  return [data]; // raw websocket: the frame IS the protocol message
}

// Outgoing commands must match the socket's framing.
function sendCmd(cmd) {
  document.dispatchEvent(new CustomEvent("psa-send", {
    detail: sockjsMode ? JSON.stringify([cmd]) : cmd,
  }));
}

function handleMessage(data) {
  if (!data.startsWith(">battle-")) return;
  const nl = data.indexOf("\n");
  const room = data.slice(1, nl === -1 ? data.length : nl).trim();
  const body = nl === -1 ? "" : data.slice(nl + 1);
  const st = rooms[room] || (rooms[room] = { lines: [], request: null });
  for (const line of body.split("\n")) {
    if (!line.startsWith("|")) continue;
    if (line.startsWith("|request|")) {
      const raw = line.slice(9).trim();
      if (!raw) continue;
      try { st.request = JSON.parse(raw); } catch { continue; }
      onRequest(room, st);
    } else {
      st.lines.push(line);
      if (!st.request && line.startsWith("|turn|")) {
        setStatus(room + " — watching (no decision request yet)");
      }
      if (line.startsWith("|win|") || line.startsWith("|tie|")) {
        setStatus(room + " — battle ended");
      }
    }
  }
}

function onRequest(room, st) {
  const req = st.request;
  if (!req || req.wait) return;
  setStatus(room + " — thinking…");
  chrome.runtime.sendMessage(
    { type: "advise", log: st.lines.join("\n"), request: JSON.stringify(req), mode: settings.mode },
    (res) => {
      if (chrome.runtime.lastError || !res || !res.ok) {
        setStatus("bridge offline — start run_assistant.bat");
        return;
      }
      renderAdvice(room, res);
      if (settings.auto && res.choose && req.rqid !== undefined) {
        // human-ish pause: usually 1-7 s, with an occasional (5%) long think of
        // 10-15 s — real players sometimes stop and stare at a position
        const delay = Math.random() < 0.05
          ? 10000 + Math.random() * 5000
          : 1000 + Math.random() * 6000;
        setTimeout(() => {
          if (rooms[room] && rooms[room].request === req) { // still the live decision
            sendCmd(room + "|/choose " + res.choose + "|" + req.rqid);
            setStatus(room + " — auto: " + res.choose);
          }
        }, delay);
      }
    }
  );
}

// ---- panel -------------------------------------------------------------------

function panel() {
  let el = document.getElementById("psa-panel");
  if (el) return el;
  el = document.createElement("div");
  el.id = "psa-panel";
  el.innerHTML =
    '<div id="psa-head">Win-Prob Assistant <span id="psa-winprob"></span></div>' +
    '<div id="psa-status">no socket yet — refresh the Showdown tab</div>' +
    '<div id="psa-best"></div><div id="psa-table"></div><div id="psa-opp"></div>' +
    '<div id="psa-controls">' +
    '<label><input type="radio" name="psa-mode" value="fast"> Fast</label> ' +
    '<label><input type="radio" name="psa-mode" value="deep"> Deep</label> ' +
    '<label id="psa-auto-label"><input type="checkbox" id="psa-auto"> Auto-play</label>' +
    "</div>";
  document.body.appendChild(el);
  el.querySelectorAll('input[name="psa-mode"]').forEach((r) =>
    r.addEventListener("change", () => {
      settings.mode = r.value;
      chrome.storage.local.set({ psaMode: r.value });
    })
  );
  el.querySelector("#psa-auto").addEventListener("change", (e) => {
    settings.auto = e.target.checked;
    chrome.storage.local.set({ psaAuto: settings.auto });
  });
  return el;
}

function renderControls() {
  const el = panel();
  el.querySelector(`input[name="psa-mode"][value="${settings.mode}"]`).checked = true;
  el.querySelector("#psa-auto").checked = settings.auto;
}

function setStatus(text) {
  panel().querySelector("#psa-status").textContent = text;
}

function renderAdvice(room, res) {
  const el = panel();
  if (typeof res.winprob === "number") {
    el.querySelector("#psa-winprob").textContent = Math.round(res.winprob * 100) + "% win";
  }
  setStatus(room + (res.mode ? " — " + res.mode : ""));
  const best = (res.table && res.table[0]) || null;
  el.querySelector("#psa-best").textContent =
    res.choose ? "▶ " + (best ? best.action : res.choose) + "   (/choose " + res.choose + ")" : "";
  const rows = (res.table || []).slice(0, 4).map((r) => {
    const wc = r.worst_case !== undefined ? Math.round(r.worst_case * 100) + "%" : "";
    return "<tr><td>" + r.action + "</td><td>" + wc + "</td></tr>";
  });
  el.querySelector("#psa-table").innerHTML = rows.length
    ? "<table><tr><th>action</th><th>worst</th></tr>" + rows.join("") + "</table>"
    : "";
  const opp = (res.opp_pred || [])
    .map((p) => (p.kind === "switch" ? "switch " + p.name : p.name) + " " + Math.round(p.prob * 100) + "%")
    .join(" · ");
  el.querySelector("#psa-opp").textContent = opp ? "them: " + opp : "";
}

panel();
