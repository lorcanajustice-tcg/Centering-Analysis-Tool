"""Disk cache for reference data (card databases, official renders)."""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path

USER_AGENT = "centering-analyzer/0.1"


def default_cache_dir() -> Path:
    env = os.environ.get("CENTERING_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "centering"


class DiskCache:
    def __init__(self, cache_dir: Path | str | None = None):
        self.dir = Path(cache_dir) if cache_dir else default_cache_dir()
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, url: str, suffix: str = "") -> Path:
        h = hashlib.sha1(url.encode()).hexdigest()[:20]
        return self.dir / (h + suffix)

    def fetch(self, url: str, ttl_days: float | None = None,
              suffix: str = "") -> Path:
        """Download url to cache (or reuse a fresh copy); returns local path."""
        p = self._path(url, suffix)
        if p.exists():
            if ttl_days is None:
                return p
            if (time.time() - p.stat().st_mtime) < ttl_days * 86400:
                return p
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        data = urllib.request.urlopen(req, timeout=120).read()
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(p)
        return p

    def fetch_json_maybe_zipped(self, url: str, ttl_days: float = 7.0) -> dict:
        """Fetch a JSON document; if url ends with .zip, unzip the single
        member. Used for lorcanajson's allCards.json.zip."""
        p = self.fetch(url, ttl_days=ttl_days,
                       suffix=".zip" if url.endswith(".zip") else ".json")
        raw = p.read_bytes()
        if url.endswith(".zip"):
            z = zipfile.ZipFile(io.BytesIO(raw))
            raw = z.read(z.namelist()[0])
        return json.loads(raw)
