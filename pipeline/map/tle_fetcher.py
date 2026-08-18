"""
pipeline/map/tle_fetcher.py — TLE Ingestion with caching and fallback

Fetches Two-Line Element sets from CelesTrak (default, no auth) or
Space-Track.org (optional, requires credentials in .env).

Cache strategy
--------------
1. If a local cache exists and is less than 24 hours old → use cache
2. Otherwise → fetch from remote source
3. If fetch fails AND cache exists (even stale) → warn and use stale cache
4. If fetch fails AND no cache → use committed fallback CSV

Fallback staleness
------------------
TLE epochs drift over time. An old epoch causes SGP4 to silently produce NaN
position vectors, making all objects fall outside the valid LEO bands.

To prevent this, _load_fallback() checks the snapshot's generation timestamp
(embedded in its # Generated: header). If it is older than
FALLBACK_MAX_AGE_DAYS (default 7), it automatically calls
generate_fallback_snapshot() to regenerate the file with a fresh epoch before
loading it. This happens at most once per startup — subsequent calls reuse the
regenerated file.

Fallback CSV format
-------------------
The committed fallback at data/tle_snapshot_fallback.csv is a plain TLE text
file (not a CSV despite the extension) — two-line elements separated by
newlines, with optional name lines.  This matches the format returned by
CelesTrak so the same parser handles both.

CelesTrak GP endpoint
---------------------
https://celestrak.org/SOCRATES/query.php  — conjunction data (not used here)
https://celestrak.org/pub/TLE/catalog.txt — full catalog (default)

No authentication required.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone, timedelta
from typing import List

import requests

# ── Configuration ──────────────────────────────────────────────────────────────
CELESTRAK_URL   = "https://celestrak.org/pub/TLE/catalog.txt"
SPACETRACK_LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
SPACETRACK_QUERY_URL = (
    "https://www.space-track.org/basicspacedata/query/class/gp/"
    "EPOCH/%3Enow-30/MEAN_MOTION/%3E11.25/ECCENTRICITY/%3C0.25/"
    "orderby/NORAD_CAT_ID/format/tle"
)

CACHE_MAX_AGE_HOURS  = 24
FALLBACK_MAX_AGE_DAYS = 7    # regenerate the synthetic snapshot if older than this
REQUEST_TIMEOUT_S    = 30

_DEFAULT_CACHE_PATH = "data/cache/tle_cache.txt"
_DEFAULT_META_PATH  = "data/cache/tle_cache_meta.json"
_DEFAULT_FALLBACK   = "data/tle_snapshot_fallback.csv"

# Compiled once — matches "# Generated: 2026-08-18T04:17:18Z" in snapshot header
_GENERATED_RE = re.compile(r"#\s*Generated:\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)")


class TLEFetcher:
    """Fetches and caches TLE data from CelesTrak or Space-Track.

    Parameters
    ----------
    data_source : str
        "celestrak" (default) or "spacetrack"
    spacetrack_user : str
        Space-Track username (only needed when data_source="spacetrack")
    spacetrack_pass : str
        Space-Track password (only needed when data_source="spacetrack")
    cache_path : str
        Path to write/read the local TLE cache file.
    cache_meta_path : str
        Path to write/read the cache metadata (timestamp) JSON.
    fallback_path : str
        Path to the committed fallback TLE snapshot.
    """

    def __init__(
        self,
        data_source: str = "celestrak",
        spacetrack_user: str = "",
        spacetrack_pass: str = "",
        cache_path: str = _DEFAULT_CACHE_PATH,
        cache_meta_path: str = _DEFAULT_META_PATH,
        fallback_path: str = _DEFAULT_FALLBACK,
    ):
        self.data_source      = data_source.lower()
        self.spacetrack_user  = spacetrack_user
        self.spacetrack_pass  = spacetrack_pass
        self.cache_path       = cache_path
        self.cache_meta_path  = cache_meta_path
        self.fallback_path    = fallback_path

    # ── Public API ─────────────────────────────────────────────────────────────

    def fetch(self, force_refresh: bool = False) -> List[str]:
        """Return TLE lines as a list of strings.

        Each element is one non-empty line from the TLE file.
        Lines are stripped of whitespace.

        Parameters
        ----------
        force_refresh : bool
            If True, bypass cache and always fetch from remote.

        Returns
        -------
        list[str]
            All non-empty lines from the TLE data source.
        """
        if not force_refresh and self._cache_is_fresh():
            return self._load_cache()

        try:
            if self.data_source == "spacetrack":
                raw = self._fetch_spacetrack()
            else:
                raw = self._fetch_celestrak()
            self._save_cache(raw)
            return _parse_tle_lines(raw)

        except Exception as exc:
            print(f"[TLEFetcher] Fetch failed ({exc}). Falling back to cache/snapshot.")
            if os.path.exists(self.cache_path):
                return self._load_cache()
            return self._load_fallback()

    def last_updated(self) -> str | None:
        """Return ISO-format timestamp of the last successful fetch, or None."""
        if not os.path.exists(self.cache_meta_path):
            return None
        with open(self.cache_meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("fetched_at")

    # ── Private: fetch from sources ───────────────────────────────────────────

    def _fetch_celestrak(self) -> str:
        print(f"[TLEFetcher] Fetching from CelesTrak...")
        resp = requests.get(CELESTRAK_URL, timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
        return resp.text

    def _fetch_spacetrack(self) -> str:
        print("[TLEFetcher] Logging in to Space-Track.org...")
        session = requests.Session()
        login_payload = {
            "identity": self.spacetrack_user,
            "password": self.spacetrack_pass,
        }
        resp = session.post(SPACETRACK_LOGIN_URL, data=login_payload,
                            timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
        print("[TLEFetcher] Fetching TLE data from Space-Track...")
        resp = session.get(SPACETRACK_QUERY_URL, timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
        return resp.text

    # ── Private: cache helpers ────────────────────────────────────────────────

    def _cache_is_fresh(self) -> bool:
        if not os.path.exists(self.cache_path):
            return False
        if not os.path.exists(self.cache_meta_path):
            return False
        try:
            with open(self.cache_meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            fetched_at = datetime.fromisoformat(meta["fetched_at"])
            age_hours = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600
            return age_hours < CACHE_MAX_AGE_HOURS
        except Exception:
            return False

    def _save_cache(self, raw_text: str) -> None:
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            f.write(raw_text)
        meta = {"fetched_at": datetime.now(timezone.utc).isoformat()}
        with open(self.cache_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        print(f"[TLEFetcher] Cache updated at {meta['fetched_at']}")

    def _load_cache(self) -> List[str]:
        with open(self.cache_path, encoding="utf-8") as f:
            raw = f.read()
        lines = _parse_tle_lines(raw)
        print(f"[TLEFetcher] Loaded {len(lines)} TLE lines from cache.")
        return lines

    def _load_fallback(self) -> List[str]:
        """Load the committed fallback snapshot, auto-regenerating it if stale."""
        self._refresh_fallback_if_stale()
        print(f"[TLEFetcher] Using committed fallback snapshot: {self.fallback_path}")
        with open(self.fallback_path, encoding="utf-8") as f:
            raw = f.read()
        return _parse_tle_lines(raw)

    def _fallback_age_days(self) -> float:
        """Return age of the fallback snapshot in days, or infinity if unreadable."""
        if not os.path.exists(self.fallback_path):
            return float("inf")
        try:
            with open(self.fallback_path, encoding="utf-8") as f:
                # Only scan the first 10 lines — the header is at the top
                for _ in range(10):
                    line = f.readline()
                    if not line:
                        break
                    m = _GENERATED_RE.search(line)
                    if m:
                        generated_at = datetime.strptime(
                            m.group(1), "%Y-%m-%dT%H:%M:%SZ"
                        ).replace(tzinfo=timezone.utc)
                        return (datetime.now(timezone.utc) - generated_at).total_seconds() / 86400.0
        except Exception:
            pass
        return float("inf")

    def _refresh_fallback_if_stale(self) -> None:
        """Regenerate the fallback snapshot if it is older than FALLBACK_MAX_AGE_DAYS."""
        age = self._fallback_age_days()
        if age <= FALLBACK_MAX_AGE_DAYS:
            return

        print(
            f"[TLEFetcher] Fallback snapshot is {age:.1f} days old "
            f"(limit {FALLBACK_MAX_AGE_DAYS} days) — regenerating with fresh epoch…"
        )
        try:
            # Import here to avoid a circular dependency at module load time.
            # tle_fetcher.py lives at <project_root>/pipeline/map/ — go up 3 levels.
            import sys as _sys
            import os as _os
            _project_root = _os.path.dirname(
                _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
            )
            _scripts_dir = _os.path.join(_project_root, "scripts")
            if _scripts_dir not in _sys.path:
                _sys.path.insert(0, _scripts_dir)
            from generate_tle_fallback import generate_fallback_snapshot
            count = generate_fallback_snapshot(self.fallback_path, verbose=True)
            print(f"[TLEFetcher] Fallback regenerated: {count} TLE objects.")
        except Exception as exc:
            print(
                f"[TLEFetcher] WARNING: fallback regeneration failed ({exc}). "
                "Using stale snapshot — SGP4 may produce degraded results."
            )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_tle_lines(raw_text: str) -> List[str]:
    """Return a list of stripped non-empty lines from raw TLE text.

    Filters out comment lines (starting with #).
    """
    return [
        line.strip()
        for line in raw_text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
