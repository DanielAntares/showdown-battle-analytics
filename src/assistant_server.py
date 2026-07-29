"""Local HTTP bridge between the browser extension and the Python advisor.

The extension (see extension/) captures the battle stream inside the user's own
Showdown client and POSTs {log, request, mode} here; this returns the advice
table plus a ready `/choose` command. Stdlib only — no new dependencies.

Run:
    python -m src.assistant_server        (or run_assistant.bat)
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.assistant import advise_for_request
from src.predict import load_model

PORT = 8765
BOOSTER = META = None


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
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            res = advise_for_request(body.get("log", ""), body.get("request"),
                                     BOOSTER, META, mode=body.get("mode", "deep"))
        except Exception as e:  # never take the bridge down mid-battle
            res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        data = json.dumps(res, default=float).encode("utf-8")
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

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
