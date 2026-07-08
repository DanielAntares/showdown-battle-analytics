"""Teammate inference: given some revealed team members, rank the likely rest.

The model is naive-Bayes-style co-occurrence over team rosters:

    score(c | revealed R) = log P(c) + sum over r in R of [log P(r|c) - log P(r)]

where P(r|c) is shrunk toward the marginal P(r) so unseen pairs contribute no
lift instead of -infinity. Fit on training-window teams only; evaluated by
hiding part of each held-out (strictly newer) test team.

Usage:
    python -m src.teammates        # fit on train teams, evaluate on test, save
"""

import json
import math
from collections import Counter

import numpy as np
import pandas as pd

from src.common import ROOT, load_config

MODEL_PATH = ROOT / "models" / "teammates.json"
ALPHA = 5.0  # shrinkage strength toward the marginal usage rate


class TeammateModel:
    def __init__(self, n_teams: int, usage: dict, co: dict):
        self.n_teams = n_teams
        self.usage = usage
        self.co = co  # "speciesA|speciesB" (sorted) -> co-occurrence count

    @classmethod
    def fit(cls, rosters: list[list[str]]) -> "TeammateModel":
        usage, co = Counter(), Counter()
        for roster in rosters:
            mons = sorted(set(roster))
            usage.update(mons)
            for i, a in enumerate(mons):
                for b in mons[i + 1:]:
                    co[f"{a}|{b}"] += 1
        return cls(len(rosters), dict(usage), dict(co))

    def _pair(self, a: str, b: str) -> int:
        key = f"{a}|{b}" if a < b else f"{b}|{a}"
        return self.co.get(key, 0)

    def scores(self, revealed: list[str]) -> pd.Series:
        """Log-score every candidate species given the revealed teammates."""
        revealed = [r for r in revealed if r in self.usage]
        out = {}
        for c, n_c in self.usage.items():
            if c in revealed:
                continue
            score = math.log(n_c / self.n_teams)
            for r in revealed:
                p_r = self.usage[r] / self.n_teams
                p_r_given_c = (self._pair(r, c) + ALPHA * p_r) / (n_c + ALPHA)
                score += math.log(p_r_given_c) - math.log(p_r)
            out[c] = score
        return pd.Series(out).sort_values(ascending=False)

    def predict(self, revealed: list[str], top: int = 10) -> pd.DataFrame:
        s = self.scores(revealed)
        posterior = np.exp(s - s.max())
        posterior /= posterior.sum()
        return pd.DataFrame(
            {"species": s.index[:top], "relative_likelihood": posterior.values[:top]})

    def save(self) -> None:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        MODEL_PATH.write_text(json.dumps(
            {"n_teams": self.n_teams, "usage": self.usage, "co": self.co}), encoding="utf-8")

    @classmethod
    def load(cls) -> "TeammateModel":
        d = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        return cls(d["n_teams"], d["usage"], d["co"])


def load_rosters() -> tuple[list[list[str]], list[list[str]]]:
    """(train, test) rosters, split by the same date cutoff as the win-prob model."""
    cfg = load_config()
    teams = pd.read_parquet(cfg["paths"]["processed"] / "teams.parquet")
    games = pd.read_parquet(cfg["paths"]["processed"] / "games.parquet")
    keep = set(games.id[games.rating >= cfg.get("train_min_rating", 0)])
    teams = teams[teams.replay_id.isin(keep)]
    cutoff = pd.Timestamp(cfg["test_split_date"]).timestamp()
    rosters = teams.groupby(["replay_id", "side"]).agg(
        roster=("species", list), uploadtime=("uploadtime", "first"))
    is_test = rosters.uploadtime >= cutoff
    full = rosters.roster.str.len() == 6  # skip the rare short-roster games
    return (rosters.roster[~is_test & full].tolist(),
            rosters.roster[is_test & full].tolist())


def evaluate(model: TeammateModel, rosters: list[list[str]], k_revealed: int,
             baseline: pd.Series, rng: np.random.Generator) -> dict:
    hits5 = hits10 = top1 = n = 0
    n_hidden = 6 - k_revealed
    for roster in rosters:
        perm = rng.permutation(6)
        revealed = [roster[i] for i in perm[:k_revealed]]
        hidden = {roster[i] for i in perm[k_revealed:]}
        ranked = model.scores(revealed).index if model else \
            baseline[~baseline.index.isin(revealed)].index
        hits5 += len(set(ranked[:5]) & hidden)
        hits10 += len(set(ranked[:10]) & hidden)
        top1 += ranked[0] in hidden
        n += 1
    return {"recall@5": hits5 / (n * min(5, n_hidden)) if n_hidden < 5 else hits5 / (n * n_hidden),
            "recall@10": hits10 / (n * n_hidden), "top1": top1 / n}


if __name__ == "__main__":
    train, test = load_rosters()
    print(f"fitting on {len(train)} train teams; evaluating on {len(test)} test teams")
    model = TeammateModel.fit(train)
    usage_rank = pd.Series(model.usage).sort_values(ascending=False)

    for k in (1, 2, 3, 4):
        rng_m, rng_b = np.random.default_rng(7), np.random.default_rng(7)
        m = evaluate(model, test, k, usage_rank, rng_m)
        b = evaluate(None, test, k, usage_rank, rng_b)
        print(f"revealed {k}: model top1 {m['top1']:.1%} r@5 {m['recall@5']:.1%} "
              f"r@10 {m['recall@10']:.1%}   | usage-baseline top1 {b['top1']:.1%} "
              f"r@5 {b['recall@5']:.1%} r@10 {b['recall@10']:.1%}")

    model.save()
    print(f"saved to {MODEL_PATH.relative_to(ROOT)}")
