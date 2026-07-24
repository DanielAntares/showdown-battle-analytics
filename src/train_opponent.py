"""Train and evaluate the opponent-action ranker.

A LightGBM LambdaRank model ranks the candidate actions within each decision;
the top-ranked one is the prediction. Accuracy is measured honestly — only on
the candidate set an observer could have enumerated at play time (`predicted`
rows) — so a move usage never proposed counts as a miss, and reported top-1 is
directly comparable to the cheap baselines it must beat.

Usage:
    python -m src.train_opponent
"""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.common import ROOT
from src.opp_features import FEATURES

DATA = ROOT / "data" / "processed" / "opp_decisions.parquet"
MODEL = ROOT / "models" / "opp_ranker.txt"
META = ROOT / "models" / "opp_ranker_meta.json"

BANDS = [(0, 1200, "<1200"), (1200, 1500, "1200-1499"),
         (1500, 1800, "1500-1799"), (1800, 9999, "1800+")]


def _band(elo) -> str:
    if not elo or (isinstance(elo, float) and np.isnan(elo)):
        return "unknown"
    return next((lbl for lo, hi, lbl in BANDS if lo <= elo < hi), "unknown")


def _group_sizes(g: pd.Series) -> np.ndarray:
    # g must be contiguous by group; return the size of each run
    return g.groupby(g, sort=False).size().to_numpy()


def train(df: pd.DataFrame) -> lgb.Booster:
    tr = df[~df.is_test].sort_values("group")
    # hold out 8% of training GROUPS for early stopping
    groups = tr.group.unique()
    rng = np.random.default_rng(0)
    val_groups = set(rng.choice(groups, size=int(len(groups) * 0.08), replace=False))
    is_val = tr.group.isin(val_groups)
    a, b = tr[~is_val].sort_values("group"), tr[is_val].sort_values("group")

    dtrain = lgb.Dataset(a[FEATURES], label=a.is_chosen, group=_group_sizes(a.group))
    dval = lgb.Dataset(b[FEATURES], label=b.is_chosen, group=_group_sizes(b.group),
                       reference=dtrain)
    params = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [1, 3],
              "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
              "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
              "verbose": -1}
    return lgb.train(params, dtrain, num_boost_round=600, valid_sets=[dval],
                     callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])


def evaluate(df: pd.DataFrame, score: np.ndarray, label: str) -> dict:
    """Top-1/top-3/coverage over the honest (predicted-only) candidate sets.

    Vectorized: rank the chosen action within its group's predicted rows via a
    single C-level groupby-rank (a Python per-group loop over ~300k groups is far
    too slow). The chosen action's true kind/band comes from the full group, so a
    decision whose real action was never predicted still counts against its bucket
    as a miss."""
    d = df[["group", "is_chosen", "predicted", "kind", "opp_elo"]].copy()
    d["_s"] = np.asarray(score)
    chosen = d[d.is_chosen == 1].drop_duplicates("group").set_index("group")

    dp = d[d.predicted].copy()
    dp["_rank"] = dp.groupby("group")["_s"].rank(method="first", ascending=False)
    crank = dp.loc[dp.is_chosen == 1].set_index("group")["_rank"]  # covered groups

    g = pd.DataFrame({"kind": chosen["kind"], "band": chosen["opp_elo"].map(_band)})
    g["rank"] = crank.reindex(g.index)               # NaN where the action wasn't predictable
    g["cover"] = g["rank"].notna().astype(int)
    g["top1"] = (g["rank"] == 1).astype(int)
    g["top3"] = (g["rank"] <= 3).astype(int)
    return {"label": label, "n": len(g),
            "top1": g.top1.mean(), "top3": g.top3.mean(), "cover": g.cover.mean(),
            "by_band": {b: (x.top1.mean(), len(x)) for b, x in g.groupby("band")},
            "by_kind": {k: (x.top1.mean(), x.top3.mean(), len(x))
                        for k, x in g.groupby("kind")}}


def _print(res: dict):
    print(f"\n{res['label']:<22} top1={res['top1']:.3f}  top3={res['top3']:.3f}  "
          f"(coverage ceiling {res['cover']:.3f}, n={res['n']})")


def main():
    df = pd.read_parquet(DATA)
    print(f"{len(df)} rows · {df.group.nunique()} decisions · "
          f"train {int((~df.is_test).sum())} / test {int(df.is_test.sum())} rows")
    test = df[df.is_test].sort_values("group")

    booster = train(df)
    model_score = booster.predict(test[FEATURES])

    print("\n=== TEST accuracy (honest: predicted candidate set only) ===")
    baselines = {
        "usage-prior": test.m_usage_prob.to_numpy(),           # most common move, never switch
        "best-damage": test.m_dmg_proxy.to_numpy(),            # strongest attack
    }
    results = {name: evaluate(test, s, name) for name, s in baselines.items()}
    results["ranker"] = evaluate(test, model_score, "ranker (LightGBM)")
    for r in results.values():
        _print(r)

    rk = results["ranker"]
    print("\nranker top-1 by opponent Elo band:")
    for b in ["<1200", "1200-1499", "1500-1799", "1800+", "unknown"]:
        if b in rk["by_band"]:
            acc, n = rk["by_band"][b]
            print(f"  {b:<10} {acc:.3f}  (n={n})")
    print("\nranker by chosen action type:")
    for k, (t1, t3, n) in rk["by_kind"].items():
        print(f"  {k:<8} top1={t1:.3f}  top3={t3:.3f}  (n={n})")

    imp = sorted(zip(FEATURES, booster.feature_importance("gain")),
                 key=lambda x: -x[1])[:12]
    print("\ntop features (gain):", ", ".join(f"{n}={v:.0f}" for n, v in imp))

    booster.save_model(str(MODEL))
    META.write_text(json.dumps({"features": FEATURES,
                                "test_top1": rk["top1"], "test_top3": rk["top3"]}), "utf-8")
    print(f"\nsaved {MODEL.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
