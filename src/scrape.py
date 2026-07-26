"""Collect rated replays from the public Pokémon Showdown replay API.

Walks the search feed back in time (51 replays per page via the `before` cursor),
keeps games at or above the configured rating floor, and downloads each full
replay JSON. Resumable: replays already on disk are skipped, and the final
`--before` cursor is printed so the next run can continue where this one stopped.

Usage:
    python -m src.scrape                     # config.yaml defaults
    python -m src.scrape --pages 40          # bigger collection run
    python -m src.scrape --target 100000     # page until 100k replays are on disk
    python -m src.scrape --before 1783400000 # continue from a cursor
"""

import argparse
import json
import time

import requests
from tqdm import tqdm

from src.common import load_config

SEARCH_URL = "https://replay.pokemonshowdown.com/search.json"
REPLAY_URL = "https://replay.pokemonshowdown.com/{id}.json"
HEADERS = {"User-Agent": "showdown-winprob (portfolio research; low-volume, rate-limited)"}


def _get_json(session: requests.Session, url: str, params: dict | None = None, retries: int = 3):
    """GET with backoff; the replay server throws transient 404/5xx under load."""
    for attempt in range(retries + 1):
        try:
            resp = session.get(url, params=params, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError):
            if attempt == retries:
                raise
            time.sleep(5 * 2**attempt)


def fetch_page(session: requests.Session, format_id: str, before: int | None) -> list[dict]:
    params = {"format": format_id}
    if before:
        params["before"] = before
    return _get_json(session, SEARCH_URL, params)


def scrape(pages: int | None, before: int | None = None, target: int | None = None) -> None:
    cfg = load_config()
    out_dir = cfg["paths"]["raw_replays"]
    out_dir.mkdir(parents=True, exist_ok=True)
    delay = cfg["request_delay_s"]

    on_disk = len(list(out_dir.glob("*.json")))  # resumed runs count toward the target
    if target and on_disk >= target:
        print(f"Already have {on_disk} replays on disk (>= target {target}); nothing to do.")
        return
    # with a target, page until it's met (or the feed runs out); otherwise honour --pages
    page_cap = (10 ** 9 if target else (pages or cfg["pages_per_run"]))

    session = requests.Session()
    downloaded = skipped = below_floor = failed = 0
    page = 0
    hit_target = False
    while page < page_cap and not hit_target:
        try:
            entries = fetch_page(session, cfg["format"], before)
        except requests.RequestException as exc:
            print(f"\nSearch feed unavailable after retries ({exc}); stopping this run early.")
            break
        if not entries:
            print("Reached the end of the search feed.")
            break
        before = entries[-1]["uploadtime"]  # cursor for the next page

        rated = [e for e in entries if (e.get("rating") or 0) >= cfg["min_rating"]]
        below_floor += len(entries) - len(rated)

        goal = f"/{target}" if target else f" (page {page + 1}/{page_cap})"
        for entry in tqdm(rated, desc=f"{on_disk + downloaded}{goal}", unit="replay"):
            out_path = out_dir / f"{entry['id']}.json"
            if out_path.exists():
                skipped += 1
                continue
            try:
                replay = _get_json(session, REPLAY_URL.format(id=entry["id"]), retries=1)
            except (requests.RequestException, ValueError):
                failed += 1  # deleted or flaky replay: not worth stalling the run over
                continue
            out_path.write_text(json.dumps(replay), encoding="utf-8")
            downloaded += 1
            if target and on_disk + downloaded >= target:
                hit_target = True
                break
            time.sleep(delay)
        page += 1
        time.sleep(delay)

    print(
        f"Done: {downloaded} downloaded, {skipped} already on disk, "
        f"{below_floor} below the {cfg['min_rating']} rating floor, {failed} unavailable. "
        f"Total on disk: {on_disk + downloaded}."
    )
    if hit_target:
        print(f"Reached the target of {target} replays.")
    else:
        print(f"Continue further back in time with: python -m src.scrape "
              f"--before {before}" + (f" --target {target}" if target else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, default=None, help="search pages to walk (51 replays each)")
    ap.add_argument("--before", type=int, default=None, help="uploadtime cursor to start from")
    ap.add_argument("--target", type=int, default=None,
                    help="page until this many replays are on disk (resumable); ignores --pages")
    args = ap.parse_args()
    scrape(pages=args.pages, before=args.before, target=args.target)
