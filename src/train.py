"""
train.py — RGB-D クロップを 4 クラスに分類する小型 CNN の学習・評価

使い方:  python src/train.py --data data/dataset.npz --out models/sorter_cnn.pt
出力:    models/sorter_cnn.pt（重み）, outputs/train_log.json, outputs/confusion.png
  - 学習は CPU で数分程度（約 2,500 サンプル, 56×56×4ch）
  - unknown(ラベル4) は学習には使わず、「未知物体の棄却率」評価にのみ使用
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene import CLASSES

N_CLASS = len(CLASSES)
UNKNOWN = N_CLASS
REJECT_THRESH = 0.60   # softmax 最大確率がこれ未満なら「未知 → 手選別」（補助的な基準）
OOD_PERCENTILE = 97.5  # 学習データの Mahalanobis 距離の上位 2.5% を未知の閾値とする（主基準）


class SorterCNN(nn.Module):
    """入力 (B,4,56,56) → ロジット (B,4)。畳み込み3段 + 全結合。パラメータ数 ≈ 10万"""
    def __init__(self, in_ch=4, n_class=N_CLASS):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),   # 28
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),      # 14
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),      # 7
            nn.AdaptiveAvgPool2d(1),
        )
        self.embed = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 64), nn.ReLU())
        self.classifier = nn.Linear(64, n_class)
        # 未知物体検出（Mahalanobis 距離, Lee et al. 2018）用の統計量。fit_ood() で設定
        self.register_buffer("ood_mean", torch.zeros(n_class, 64))
        self.register_buffer("ood_prec", torch.eye(64))
        self.register_buffer("ood_thresh", torch.tensor(float("inf")))

    def feature(self, x):
        return self.embed(self.features(x))

    def ood_feature(self, x):
        """未知検出用の特徴量: 畳み込み最終段の平均プーリング出力 (B,64)。
        全結合層以降はクラス判別に特化して情報が潰れる（neural collapse）ため、手前の特徴を使う方が
        未知物体の棄却率が高かった（検証: 52% → 74%）"""
        return self.features(x).flatten(1)

    def forward(self, x):
        return self.classifier(self.feature(x))

    @torch.no_grad()
    def ood_distance(self, x):
        """各サンプルの特徴量と最近傍クラス平均との Mahalanobis 距離"""
        f = self.ood_feature(x)                               # (B,64)
        diff = f[:, None, :] - self.ood_mean[None, :, :]      # (B,C,64)
        d2 = torch.einsum("bci,ij,bcj->bc", diff, self.ood_prec, diff)
        return d2.min(dim=1).values.sqrt()

    @torch.no_grad()
    def fit_ood(self, X, y, percentile=OOD_PERCENTILE):
        """学習データの特徴量からクラス平均・共有共分散（縮小推定）・棄却閾値を推定"""
        self.eval()
        f = self.ood_feature(X)
        C = self.classifier.out_features
        means = torch.stack([f[y == c].mean(0) for c in range(C)])
        centered = f - means[y]
        cov = centered.T @ centered / len(f)
        cov = 0.9 * cov + 0.1 * torch.eye(cov.shape[0]) * cov.diagonal().mean()   # 縮小推定
        self.ood_mean.copy_(means); self.ood_prec.copy_(torch.linalg.inv(cov))
        d = self.ood_distance(X)
        self.ood_thresh.fill_(float(np.percentile(d.numpy(), percentile)))


def augment(x, rng):
    """学習時データ拡張: ランダム平行移動(±4px)・左右/上下反転・明るさ揺らぎ。x: (B,4,H,W) tensor"""
    B, C, H, W = x.shape
    out = torch.empty_like(x)
    for i in range(B):
        dx, dy = rng.integers(-4, 5, 2)
        xi = torch.roll(x[i], shifts=(int(dy), int(dx)), dims=(1, 2))
        if rng.random() < 0.5:
            xi = torch.flip(xi, dims=[2])
        if rng.random() < 0.5:
            xi = torch.flip(xi, dims=[1])
        gain = 1.0 + rng.normal(0, 0.08)
        xi = xi.clone()
        xi[:3] = (xi[:3] * gain).clamp(0, 1)
        out[i] = xi
    return out


def predict(model, x, thresh=REJECT_THRESH, use_ood=True, ood_scale=1.0):
    """推論: (ラベル, 確率, 全クラス確率, OOD距離) を返す。
    最大確率 < thresh または Mahalanobis 距離 > 閾値×ood_scale の場合ラベルは UNKNOWN。"""
    model.eval()
    with torch.no_grad():
        prob = F.softmax(model(x), dim=1)
        dist = model.ood_distance(x)
    p, lab = prob.max(dim=1)
    unk = p < thresh
    if use_ood:
        unk |= dist > model.ood_thresh * ood_scale
    lab = torch.where(unk, torch.full_like(lab, UNKNOWN), lab)
    return lab.numpy(), p.numpy(), prob.numpy(), dist.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.npz")
    ap.add_argument("--out", default="models/sorter_cnn.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    z = np.load(a.data)
    X, y, sc = z["X"].astype(np.float32), z["y"], z["scene"]
    # シーン単位で学習/検証を分割（同じシーンの物体が両方に入るリークを防ぐ）
    scenes = np.unique(sc)
    val_scenes = set(rng.choice(scenes, size=int(len(scenes) * 0.2), replace=False).tolist())
    is_val = np.array([s in val_scenes for s in sc])
    known = y < N_CLASS
    tr = ~is_val & known
    va = is_val & known
    un = is_val & ~known
    Xtr, ytr = torch.tensor(X[tr]), torch.tensor(y[tr])
    Xva, yva = torch.tensor(X[va]), torch.tensor(y[va])
    Xun = torch.tensor(X[un])
    print(f"train {len(Xtr)}  val {len(Xva)}  unknown(val) {len(Xun)}  class counts {np.bincount(y[tr])}")

    model = SorterCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    log = []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot, correct, n = 0.0, 0, 0
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            xb = augment(Xtr[idx], rng); yb = ytr[idx]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx); correct += (logits.argmax(1) == yb).sum().item(); n += len(idx)
        sched.step()
        lab, p, _, _ = predict(model, Xva, thresh=0.0, use_ood=False)
        val_acc = float((lab == yva.numpy()).mean())
        log.append(dict(epoch=ep + 1, loss=tot / n, train_acc=correct / n, val_acc=val_acc))
        print(f"ep {ep+1:2d} loss {tot/n:.3f} train_acc {correct/n:.3f} val_acc {val_acc:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # ---- 未知物体検出器のフィッティング（学習データのみ使用）----
    model.fit_ood(Xtr, ytr)
    print(f"OOD threshold (Mahalanobis) = {model.ood_thresh.item():.2f}")
    # ---- 最終評価（棄却閾値あり）----
    lab, p, prob, dist = predict(model, Xva)
    yv = yva.numpy()
    cm = np.zeros((N_CLASS, N_CLASS + 1), int)
    for t, l in zip(yv, lab):
        cm[t, l] += 1
    acc_known = float((lab == yv).mean())
    reject_known = float((lab == UNKNOWN).mean())
    lab_u, p_u, _, dist_u = predict(model, Xun)
    reject_unknown = float((lab_u == UNKNOWN).mean()) if len(Xun) else float("nan")
    # softmax のみでの棄却性能（比較用）
    lab_s, _, _, _ = predict(model, Xva, 0.8, use_ood=False); lab_su, _, _, _ = predict(model, Xun, 0.8, use_ood=False)
    softmax_only = dict(thresh=0.8, acc_known=float((lab_s == yv).mean()), reject_unknown=float((lab_su == UNKNOWN).mean()))
    # 閾値スイープ（Mahalanobis 距離の倍率, レポート用）
    sweep = []
    for sc in np.linspace(0.5, 2.0, 40):
        l1, _, _, _ = predict(model, Xva, ood_scale=sc); l2, _, _, _ = predict(model, Xun, ood_scale=sc)
        sweep.append(dict(scale=float(sc), thresh=float(model.ood_thresh.item() * sc), acc_known=float((l1 == yv).mean()),
                          reject_known=float((l1 == UNKNOWN).mean()),
                          reject_unknown=float((l2 == UNKNOWN).mean()) if len(Xun) else None))
    result = dict(train_log=log, n_train=int(len(Xtr)), n_val=int(len(Xva)), n_unknown=int(len(Xun)),
                  reject_thresh=REJECT_THRESH, val_acc_with_reject=acc_known, val_reject_rate_known=reject_known,
                  val_reject_rate_unknown=reject_unknown, confusion=cm.tolist(),
                  classes=CLASSES + ["unknown"], threshold_sweep=sweep, softmax_only=softmax_only,
                  ood_thresh=float(model.ood_thresh.item()), ood_percentile=OOD_PERCENTILE,
                  dist_known_val=dist.tolist(), dist_unknown_val=dist_u.tolist(),
                  n_params=sum(p.numel() for p in model.parameters()))
    print(json.dumps({k: v for k, v in result.items() if k not in ("train_log", "threshold_sweep")}, indent=1, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(model.state_dict(), a.out)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/train_log.json", "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    plot(result)


def plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp); plt.rcParams["font.family"] = "Noto Sans CJK JP"
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ep = [l["epoch"] for l in result["train_log"]]
    ax[0].plot(ep, [l["train_acc"] for l in result["train_log"]], label="学習")
    ax[0].plot(ep, [l["val_acc"] for l in result["train_log"]], label="検証")
    ax[0].set_xlabel("エポック"); ax[0].set_ylabel("正解率"); ax[0].set_ylim(0.5, 1.02); ax[0].legend(); ax[0].set_title("学習曲線")
    cm = np.array(result["confusion"]); cls = result["classes"]
    im = ax[1].imshow(cm, cmap="Blues")
    ax[1].set_xticks(range(len(cls))); ax[1].set_xticklabels(cls, rotation=30)
    ax[1].set_yticks(range(len(cls) - 1)); ax[1].set_yticklabels(cls[:-1])
    ax[1].set_xlabel("予測"); ax[1].set_ylabel("正解"); ax[1].set_title("混同行列（検証, 棄却あり）")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax[1].text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    dk, du = result["dist_known_val"], result["dist_unknown_val"]
    bins = np.linspace(0, max(max(dk), max(du)) * 1.05, 40)
    ax[2].hist(dk, bins=bins, alpha=0.6, label="既知4クラス（検証）")
    ax[2].hist(du, bins=bins, alpha=0.6, label="未知物体（検証）")
    ax[2].axvline(result["ood_thresh"], color="red", ls=":", label=f"棄却閾値 {result['ood_thresh']:.1f}")
    ax[2].set_xlabel("特徴空間の Mahalanobis 距離"); ax[2].set_ylabel("サンプル数"); ax[2].legend(); ax[2].set_title("未知物体の棄却（OOD検出）")
    fig.tight_layout(); fig.savefig("outputs/training_summary.png", dpi=130)


if __name__ == "__main__":
    main()
