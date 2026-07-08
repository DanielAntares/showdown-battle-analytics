"""Train and evaluate the win-probability models.

Three models of increasing knowledge, all evaluated on strictly-newer games:
  1. elo-only     — logistic regression on the pre-game rating difference
  2. state-lr     — logistic regression on a handful of battle-state differentials
  3. lightgbm     — gradient boosting on the full turn snapshot

Win probability is a calibration problem, so the headline metrics are log loss
and Brier score (reliability diagram saved to reports/figures/), with AUC as the
ranking sanity check.

Usage:
    python -m src.train
"""

import json

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common import ROOT
from src.features import CATEGORICAL, build_features, load_raw, mirror_raw, time_split

STATE_LR_FEATURES = ["rating_diff", "hp_diff", "fainted_diff", "turn",
                     "p1_active_hp", "p2_active_hp"]
TURN_BUCKETS = [1, 6, 11, 16, 21, 31, 1000]

# chart chrome + categorical slots from the validated reference palette (dataviz skill)
SURFACE, GRID, BASELINE, INK, INK_2, MUTED = (
    "#fcfcfb", "#e1e0d9", "#c3c2b7", "#0b0b0b", "#52514e", "#898781")
BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"  # lgbm, state-lr, elo-only


def evaluate(name: str, y_true, p) -> dict:
    return {
        "model": name,
        "log_loss": log_loss(y_true, p),
        "brier": brier_score_loss(y_true, p),
        "auc": roc_auc_score(y_true, p),
    }


def style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def make_figure(test: pd.DataFrame, preds: dict[str, np.ndarray]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    y = test.label_p1_win

    ax1.plot([0, 1], [0, 1], color=BASELINE, linewidth=1.2, linestyle="--", zorder=1)
    for name, color in (("lightgbm", BLUE), ("state-lr", AQUA)):
        frac, mean_pred = calibration_curve(y, preds[name], n_bins=15, strategy="quantile")
        ax1.plot(mean_pred, frac, color=color, linewidth=2, marker="o", markersize=4.5,
                 label=name, zorder=3)
        ax1.annotate(name, (mean_pred[-1], frac[-1]), xytext=(6, -2 if name == "lightgbm" else -14),
                     textcoords="offset points", fontsize=9, color=INK_2)
    ax1.set_title("Reliability on held-out weeks", fontsize=11, loc="left")
    ax1.set_xlabel("predicted P1 win probability")
    ax1.set_ylabel("observed P1 win frequency")
    ax1.legend(frameon=False, loc="upper left", labelcolor=INK_2, fontsize=9)

    buckets = pd.cut(test.turn, bins=TURN_BUCKETS, right=False)
    labels = [f"{b.left}–{b.right - 1}" if b.right < 100 else f"{b.left}+"
              for b in buckets.cat.categories]
    for name, color in (("lightgbm", BLUE), ("elo-only", YELLOW)):
        by_bucket = [log_loss(y[buckets == b], preds[name][buckets == b])
                     for b in buckets.cat.categories]
        ax2.plot(labels, by_bucket, color=color, linewidth=2, marker="o", markersize=4.5,
                 label=name)
        ax2.annotate(name, (len(labels) - 1, by_bucket[-1]), xytext=(6, -3),
                     textcoords="offset points", fontsize=9, color=INK_2)
    ax2.set_title("Log loss by game phase", fontsize=11, loc="left")
    ax2.set_xlabel("turn")
    ax2.set_ylabel("log loss (lower is better)")
    ax2.set_xlim(-0.4, len(labels) + 0.9)
    ax2.legend(frameon=False, loc="lower left", labelcolor=INK_2, fontsize=9)

    for ax in (ax1, ax2):
        style_axis(ax)
    fig.tight_layout()
    out = ROOT / "reports" / "figures" / "calibration.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"\nFigure saved to {out.relative_to(ROOT)}")


def main() -> None:
    raw = load_raw()
    is_test = time_split(raw)
    df, features, levels = build_features(raw)
    train, test = df[~is_test], df[is_test]
    print(f"train: {train.replay_id.nunique()} games / {len(train)} rows | "
          f"test: {test.replay_id.nunique()} games / {len(test)} rows")

    # hold out 10% of train games for early stopping
    rng = np.random.default_rng(42)
    game_ids = train.replay_id.unique()
    val_ids = set(rng.choice(game_ids, size=len(game_ids) // 10, replace=False))
    is_val = train.replay_id.isin(val_ids)
    fit, val = train[~is_val], train[is_val]

    # mirror augmentation: every fitted position also seen from the other seat
    # (label flipped) — doubles the sample and enforces p1/p2 symmetry
    mirror, _, _ = build_features(mirror_raw(raw[~is_test][~is_val.values]), levels=levels)
    fit = pd.concat([fit, mirror], ignore_index=True)

    preds: dict[str, np.ndarray] = {}
    y_test = test.label_p1_win

    preds["always-0.5"] = np.full(len(test), 0.5)

    elo = make_pipeline(StandardScaler(), LogisticRegression())
    elo.fit(train[["rating_diff"]], train.label_p1_win)
    preds["elo-only"] = elo.predict_proba(test[["rating_diff"]])[:, 1]

    state_lr = make_pipeline(StandardScaler(), LogisticRegression())
    state_lr.fit(train[STATE_LR_FEATURES], train.label_p1_win)
    preds["state-lr"] = state_lr.predict_proba(test[STATE_LR_FEATURES])[:, 1]

    model = lgb.LGBMClassifier(
        n_estimators=6000,
        learning_rate=0.03,
        num_leaves=127,
        min_child_samples=50,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        fit[features], fit.label_p1_win,
        eval_set=[(val[features], val.label_p1_win)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    preds["lightgbm"] = model.predict_proba(test[features])[:, 1]
    print(f"lightgbm stopped at {model.best_iteration_} trees")

    results = pd.DataFrame([evaluate(n, y_test, p) for n, p in preds.items()])
    print("\n" + results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    imp = pd.Series(model.booster_.feature_importance("gain"), index=features)
    print("\nTop features by gain:")
    print((imp.sort_values(ascending=False).head(12) / imp.sum()).to_string(
        float_format=lambda x: f"{x:.1%}"))

    out_dir = ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    model.booster_.save_model(out_dir / "winprob_lgbm.txt")
    meta = {
        "features": features,
        "categories": {c: levels[c] for c in CATEGORICAL},
        "best_iteration": model.best_iteration_,
        "test_metrics": results.set_index("model").loc["lightgbm"].to_dict(),
    }
    (out_dir / "feature_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nModel saved to models/winprob_lgbm.txt (+ feature_meta.json)")

    make_figure(test, preds)


if __name__ == "__main__":
    main()
