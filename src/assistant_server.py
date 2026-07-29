"""Local HTTP bridge between the browser extension and the Python advisor.

The extension (see extension/) captures the battle stream inside the user's own
Showdown client and POSTs {log, request, mode} here; this returns the advice
table plus a ready `/choose` command. Stdlib only — no new dependencies.

Run:
    python -m src.assistant_server        (or run_assistant.bat)
"""

import json
import math
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.assistant import advise_for_request
from src.predict import load_model

PORT = 8765
BOOSTER = META = None


def _safe(obj):
    """Coerce an advice payload into strictly JSON-legal values: numpy scalars ->
    Python, arrays/sets -> lists, NaN/inf -> null. A single stray numpy value or NaN
    made json.dumps emit invalid JSON (or raise, killing the reply) — which the
    browser reported as 'bridge offline (no response)'."""
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_safe(v) for v in obj]
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return [_safe(v) for v in obj.tolist()]
        if isinstance(obj, np.generic):
            return _safe(obj.item())
    except Exception:
        pass
    try:
        f = float(obj)
        return f if math.isfinite(f) else None
    except Exception:
        return str(obj)


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        # a public site (play.pokemonshowdown.com) reaching a loopback address now
        # needs this on the preflight, or Chrome's Private Network Access blocks it
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        if self.path != "/advise":
            self.send_response(404)
            self._cors()
            self.end_headers()
            return
        try:  # serialization is INSIDE the guard now — nothing here can kill the reply
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            print(f"-> advise (mode={body.get('mode', 'deep')}, {len(body.get('log', ''))} log chars)",
                  flush=True)
            res = advise_for_request(body.get("log", ""), body.get("request"),
                                     BOOSTER, META, mode=body.get("mode", "deep"))
            print(f"  ok: choose={res.get('choose')} winprob={res.get('winprob')}", flush=True)
            data = json.dumps(_safe(res)).encode("utf-8")
        except Exception as e:  # never take the bridge down mid-battle
            traceback.print_exc()
            data = json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}).encode("utf-8")
        try:
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            traceback.print_exc()  # client hung up; log and move on

    def log_message(self, fmt, *args):  # keep the console readable
        pass


def main():
    global BOOSTER, META
    print("loading model…")
    BOOSTER, META = load_model()
    print(f"assistant bridge on http://127.0.0.1:{PORT}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
