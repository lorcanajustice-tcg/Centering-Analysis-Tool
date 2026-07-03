"""Local web UI for the centering analyzer.

Stdlib-only HTTP server (no Flask needed): serves web/index.html and an
/api/analyze endpoint. Photos arrive as base64 JSON from the browser, results
(JSON + full-res overlays) are saved under results/<timestamp>/.

Run:  python webapp.py   then open http://127.0.0.1:8737/
"""
from __future__ import annotations

import base64
import datetime
import io
import json
import sys
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

PORT = 8737
_lock = threading.Lock()  # one analysis at a time (CPU-bound anyway)


def _overlay_data_url(path: str, max_w: int = 900) -> str:
    import cv2
    img = cv2.imread(path)
    if img is None:
        return ""
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def run_analysis(payload: dict) -> dict:
    from centering import analyze_back, analyze_borderless, analyze_card
    from centering.games.lorcana import LORCANA

    mode = payload.get("mode")
    card_id = (payload.get("card_id") or "").strip()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        paths = {}
        for role in ("back", "front"):
            ph = payload.get("photos", {}).get(role)
            if ph:
                p = Path(td) / (role + "_" + Path(ph["name"]).name)
                p.write_bytes(base64.b64decode(ph["data"]))
                paths[role] = p

        if mode == "back":
            if "back" not in paths:
                raise ValueError("select a back photo")
            res = analyze_back(paths["back"], LORCANA, out_dir=out_dir)
            faces = {"back": res}
            result = res.to_dict()
        elif mode == "front":
            if "front" not in paths:
                raise ValueError("select a front photo")
            if not card_id:
                raise ValueError("card ID is required for a borderless front "
                                 '(e.g. "6/C2", "7:69", or a unique card name)')
            res = analyze_borderless(paths["front"], card_id, LORCANA,
                                     out_dir=out_dir)
            faces = {"front": res}
            result = res.to_dict()
        elif mode == "card":
            if "back" not in paths or "front" not in paths:
                raise ValueError("combined mode needs both photos")
            if not card_id:
                raise ValueError("card ID is required for the front analysis")
            res = analyze_card(back_photo=paths["back"],
                               front_photo=paths["front"],
                               card_id=card_id, game=LORCANA, out_dir=out_dir)
            faces = {"back": res.back, "front": res.front}
            result = res.to_dict()
        else:
            raise ValueError(f"unknown mode {mode!r}")

    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    overlays = {}
    for role, face in faces.items():
        if face is not None and face.overlay:
            overlays[role] = {"preview": _overlay_data_url(face.overlay),
                              "path": str(face.overlay)}
    return {"result": result, "overlays": overlays,
            "saved_to": str(out_dir)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = (ROOT / "web" / "index.html").read_bytes()
            self._send(200, page, "text/html; charset=utf-8")
        elif self.path == "/api/health":
            self._send(200, b'{"ok": true}', "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/analyze":
            self._send(404, b"not found", "text/plain")
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n))
            with _lock:
                out = run_analysis(payload)
            self._send(200, json.dumps(out).encode(), "application/json")
        except ValueError as e:
            self._send(400, json.dumps({"error": str(e)}).encode(),
                       "application/json")
        except Exception as e:
            traceback.print_exc()
            self._send(500, json.dumps(
                {"error": f"{type(e).__name__}: {e}"}).encode(),
                "application/json")


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Centering Analyzer running at http://127.0.0.1:{PORT}/")
    print("Close this window (or Ctrl+C) to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
