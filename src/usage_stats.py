"""Download Smogon monthly usage stats (the detailed 'chaos' JSON).

These provide usage rates, movesets, items, abilities, and teammate
co-occurrence — inputs for matchup features (Phase 3) and the team-inference
stretch goal (Phase 5).

Usage:
    python -m src.usage_stats                # month/baseline from config.yaml
    python -m src.usage_stats --month 2026-05
"""

import argparse

import requests

from src.common import load_config

CHAOS_URL = "https://www.smogon.com/stats/{month}/chaos/{format}-{baseline}.json"


def fetch(month: str | None = None, baseline: int | None = None) -> None:
    cfg = load_config()
    month = month or cfg["stats_month"]
    baseline = baseline or cfg["stats_baseline"]
    out_dir = cfg["paths"]["raw_stats"]
    out_dir.mkdir(parents=True, exist_ok=True)

    url = CHAOS_URL.format(month=month, format=cfg["format"], baseline=baseline)
    out_path = out_dir / f"{cfg['format']}-{month}-{baseline}.json"
    if out_path.exists():
        print(f"Already downloaded: {out_path.name}")
        return
    print(f"Fetching {url} ...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    print(f"Saved {out_path.name} ({len(resp.content) / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--month", default=None, help="YYYY-MM (default: config.yaml)")
    ap.add_argument("--baseline", type=int, default=None, help="0/1500/1695/1825")
    args = ap.parse_args()
    fetch(args.month, args.baseline)
