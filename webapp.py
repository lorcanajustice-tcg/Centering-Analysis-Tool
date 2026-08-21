"""The little web page you use to check cards.

This runs a small web server on your own computer - nothing is sent
anywhere except the one request that fetches the official picture of the
card. It serves web/index.html, takes the photos you drop in, runs the
measurement, and saves the results and the marked-up pictures into
results/<date and time>/.

To run it by hand:  python webapp.py   then open http://127.0.0.1:8737/
(On Windows, double-click run_analyzer.bat instead.)
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

        mb = payload.get("manual_bbox")
        mb = tuple(float(v) for v in mb) if mb else None
        if mode == "back":
            if "back" not in paths:
                raise ValueError("Add a photo of the back of the card.")
            res = analyze_back(paths["back"], LORCANA, out_dir=out_dir)
            faces = {"back": res}
            result = res.to_dict()
        elif mode == "front":
            if "front" not in paths:
                raise ValueError("Add a photo of the front of the card.")
            if not card_id:
                raise ValueError(
                    "Say which card this is so the official picture can be "
                    'looked up - for example "8-210", or part of the '
                    "card's name.")
            res = analyze_borderless(paths["front"], card_id, LORCANA,
                                     out_dir=out_dir, manual_bbox=mb)
            faces = {"front": res}
            result = res.to_dict()
        elif mode == "card":
            if "back" not in paths or "front" not in paths:
                raise ValueError("Checking the front and the back needs "
                                 "both photos.")
            if not card_id:
                raise ValueError(
                    "Say which card this is so the official picture can be "
                    'looked up - for example "8-210".')
            res = analyze_card(back_photo=paths["back"],
                               front_photo=paths["front"],
                               card_id=card_id, game=LORCANA, out_dir=out_dir,
                               front_manual_bbox=mb)
            faces = {"back": res.back, "front": res.front}
            result = res.to_dict()
        else:
            raise ValueError("Choose front and back, back only, or front "
                             "only.")

    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    overlays = {}
    for role, face in faces.items():
        if face is not None and face.overlay:
            overlays[role] = {"preview": _overlay_data_url(face.overlay),
                              "path": str(face.overlay)}
    return {"result": result, "overlays": overlays,
            "saved_to": str(out_dir)}


def run_detect(payload: dict) -> dict:
    """Auto-detect the card id from a front photo (base64 JSON in)."""
    from centering.identify import detect_card_id
    from centering.cache import DiskCache

    ph = payload.get("photo") or {}
    if not ph.get("data"):
        raise ValueError("No photo of the front was sent, so the card "
                         "cannot be identified.")
    images_dir = ROOT / "card_db" / "images"
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / ("front_" + Path(ph.get("name") or "front").name)
        p.write_bytes(base64.b64decode(ph["data"]))
        res = detect_card_id(
            p, ROOT / "card_db" / "index.json",
            images_dir if images_dir.exists() else None,
            ROOT / "card_db" / "sig_index.json", cache=DiskCache())
    return res.to_dict()


def run_preview(payload: dict) -> dict:
    """Downscaled JPEG preview of an uploaded photo, via the SAME loader
    the analyzers use (EXIF orientation + HEIC), so canvas coordinates
    map 1:1 onto analysis coordinates."""
    from centering.imgio import load_photo
    import cv2
    ph = payload.get("photo") or {}
    if not ph.get("data"):
        raise ValueError("No photo was sent to show.")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / ("prev_" + Path(ph.get("name") or "img").name)
        p.write_bytes(base64.b64decode(ph["data"]))
        rgb, _gray, _inp = load_photo(p)
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    scale = min(1.0, 900.0 / w)
    if scale < 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("That photo could not be opened. Try a JPG, PNG "
                         "or HEIC file.")
    return {"preview": "data:image/jpeg;base64,"
            + base64.b64encode(buf).decode(),
            "width": w, "height": h}


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
        if self.path not in ("/api/analyze", "/api/detect",
                             "/api/preview"):
            self._send(404, b"not found", "text/plain")
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n))
            with _lock:
                out = (run_detect(payload) if self.path == "/api/detect"
                       else run_preview(payload)
                       if self.path == "/api/preview"
                       else run_analysis(payload))
            self._send(200, json.dumps(out).encode(), "application/json")
        except ValueError as e:
            self._send(400, json.dumps({"error": str(e)}).encode(),
                       "application/json")
        except Exception as e:
            traceback.print_exc()
            # The person reading this is not a programmer: give them the
            # plain sentence the code raised, and leave the technical
            # traceback in the console window (printed just above).
            self._send(500, json.dumps(
                {"error": str(e) or "Something went wrong while checking "
                 "this card. The details are in the black window this "
                 "program opened."}).encode(),
                "application/json")


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Card Centering Checker is running at http://127.0.0.1:{PORT}/")
    print("Your browser should open on its own. If not, copy that address")
    print("into your browser.")
    print("Close this window when you are finished.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
