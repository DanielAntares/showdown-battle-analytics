"""Experiment: does a sequence model (GRU) beat the snapshot LightGBM?

The production win-prob model reads a single turn's snapshot. A natural question
is whether modeling the *sequence* of turns — so the model can see momentum and
history directly — does better. This trains a GRU over per-turn feature vectors
(numeric features + embeddings for both active species) that predicts the winner
from every prefix of the game, and compares its held-out log loss / AUC to
LightGBM's on the same time-based test split.

Honest caveats (kept simple on purpose): trained on a subsample without the
mirror augmentation LightGBM uses, minor categoricals (status/weather/terrain)
dropped, CPU only. Read the result as "is there a signal worth chasing," not a
tuned head-to-head. Requires torch (not a runtime dependency of the app).

Usage:
    python -m src.seq_experiment
"""

import numpy as np
import pandas as pd

from src.features import CATEGORICAL, build_features, load_raw, time_split

TRAIN_GAMES = 6000   # subsample for CPU tractability
MAX_LEN = 45
SEED = 42


def build_sequences(df, numeric_cols, vocab, mean, std):
    """Per game: (numeric [T,F], p1 species idx [T], p2 species idx [T], label)."""
    seqs = []
    for rid, g in df.groupby("replay_id", sort=False):
        g = g.sort_values("turn").head(MAX_LEN)
        num = ((g[numeric_cols].to_numpy(dtype="float32") - mean) / std)
        num = np.nan_to_num(num)
        p1 = g.p1_active_species.astype(str).map(lambda s: vocab.get(s, 1)).to_numpy()
        p2 = g.p2_active_species.astype(str).map(lambda s: vocab.get(s, 1)).to_numpy()
        seqs.append((num, p1.astype("int64"), p2.astype("int64"),
                     int(g.label_p1_win.iloc[0])))
    return seqs


def main() -> None:
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pad_sequence
    from sklearn.metrics import log_loss, roc_auc_score

    torch.manual_seed(SEED)
    raw = load_raw()
    is_test = time_split(raw)
    df, features, levels = build_features(raw)
    numeric_cols = [f for f in features if f not in CATEGORICAL]

    train_df, test_df = df[~is_test], df[is_test]
    rng = np.random.default_rng(SEED)
    tgames = train_df.replay_id.unique()
    keep = set(rng.choice(tgames, size=min(TRAIN_GAMES, len(tgames)), replace=False))
    val_ids = set(list(keep)[: len(keep) // 10])
    fit_df = train_df[train_df.replay_id.isin(keep - val_ids)]
    val_df = train_df[train_df.replay_id.isin(val_ids)]

    species = ["<pad>", "<unk>"] + levels["p1_active_species"]
    vocab = {s: i for i, s in enumerate(species)}
    mean = fit_df[numeric_cols].to_numpy(dtype="float32").mean(0)
    std = fit_df[numeric_cols].to_numpy(dtype="float32").std(0) + 1e-6

    fit_seq = build_sequences(fit_df, numeric_cols, vocab, mean, std)
    val_seq = build_sequences(val_df, numeric_cols, vocab, mean, std)
    test_seq = build_sequences(test_df, numeric_cols, vocab, mean, std)
    print(f"sequences — fit {len(fit_seq)}, val {len(val_seq)}, test {len(test_seq)}; "
          f"{len(numeric_cols)} numeric features, {len(species)} species")

    class GRUWin(nn.Module):
        def __init__(self, n_num, n_species, emb=16, hid=64):
            super().__init__()
            self.emb = nn.Embedding(n_species, emb, padding_idx=0)
            self.gru = nn.GRU(n_num + 2 * emb, hid, batch_first=True)
            self.head = nn.Linear(hid, 1)

        def forward(self, num, p1, p2):
            x = torch.cat([num, self.emb(p1), self.emb(p2)], dim=-1)
            out, _ = self.gru(x)
            return self.head(out).squeeze(-1)

    def batches(seqs, bs=64, shuffle=True):
        order = np.random.permutation(len(seqs)) if shuffle else range(len(seqs))
        for i in range(0, len(seqs), bs):
            chunk = [seqs[j] for j in order[i:i + bs]]
            nums = pad_sequence([torch.tensor(c[0]) for c in chunk], batch_first=True)
            p1 = pad_sequence([torch.tensor(c[1]) for c in chunk], batch_first=True)
            p2 = pad_sequence([torch.tensor(c[2]) for c in chunk], batch_first=True)
            lens = torch.tensor([len(c[0]) for c in chunk])
            y = torch.tensor([c[3] for c in chunk], dtype=torch.float32)
            mask = torch.arange(nums.shape[1])[None, :] < lens[:, None]
            yield nums, p1, p2, y, mask

    model = GRUWin(len(numeric_cols), len(species))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def eval_loss(seqs):
        model.eval()
        ps, ys = [], []
        with torch.no_grad():
            for nums, p1, p2, y, mask in batches(seqs, shuffle=False):
                logit = model(nums, p1, p2)
                prob = torch.sigmoid(logit)
                ps.append(prob[mask].numpy()); ys.append(y[:, None].expand_as(mask)[mask].numpy())
        return np.concatenate(ps), np.concatenate(ys)

    best_val, best_state, patience = 1e9, None, 0
    for epoch in range(30):
        model.train()
        for nums, p1, p2, y, mask in batches(fit_seq):
            opt.zero_grad()
            logit = model(nums, p1, p2)
            target = y[:, None].expand_as(mask).float()
            loss = (bce(logit, target) * mask).sum() / mask.sum()
            loss.backward(); opt.step()
        vp, vy = eval_loss(val_seq)
        vll = log_loss(vy, np.clip(vp, 1e-6, 1 - 1e-6))
        if vll < best_val - 1e-4:
            best_val, best_state, patience = vll, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
        print(f"epoch {epoch + 1:2d}  val log loss {vll:.4f}")
        if patience >= 4:
            break

    model.load_state_dict(best_state)
    tp, ty = eval_loss(test_seq)
    tp = np.clip(tp, 1e-6, 1 - 1e-6)
    print(f"\nGRU   test log loss {log_loss(ty, tp):.4f} | auc {roc_auc_score(ty, tp):.4f}")
    print("LGBM  test log loss 0.6089 | auc 0.7233  (production, full data + mirror aug)")


if __name__ == "__main__":
    main()
