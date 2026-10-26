"""Train BrainWaveID (or the EEGNet baseline) and evaluate.

Examples
  python -m brainwaveid.train --data synthetic --out runs/smoke --epochs 5
  python -m brainwaveid.train --data deap --root /path/data_preprocessed_python --out runs/deap --cache cache/deap
  python -m brainwaveid.train --data inhouse --manifest data/manifest.csv --out runs/inhouse --cache cache/inhouse
  # ablations:  --no_lstm  --uniform_weights  --no_aug          baseline: --model eegnet
"""
import argparse, json, os, random, time, warnings
warnings.filterwarnings("ignore", message="Using padding=.same.")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .data import add_data_args, default_notch, load_recordings, prepare, prepare_dreamer
from .model import build_model, count_params
from .evaluate import evaluate_run, print_summary


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def augment(x, cfg: Config):
    """Batch augmentation on the raw window (Sec. VI-A): Gaussian noise @ SNR, time shift, channel dropout."""
    B, C, T = x.shape
    p = x.pow(2).mean(dim=(1, 2), keepdim=True)
    noise_p = p / (10 ** (cfg.aug_snr_db / 10))
    x = x + torch.randn_like(x) * noise_p.sqrt()
    max_shift = int(cfg.aug_max_shift_ms / 1000 * cfg.fs_target)
    if max_shift > 0:
        shifts = torch.randint(-max_shift, max_shift + 1, (B,))
        x = torch.stack([torch.roll(x[i], int(shifts[i]), dims=-1) for i in range(B)])   # circular shift
    keep = (torch.rand(B, C, 1, device=x.device) > cfg.aug_channel_drop).float()
    return x * keep


def run_epoch(model, loader, crit, device, cfg, opt=None):
    train = opt is not None
    model.train(train)
    tot_loss = tot_ok = n = 0
    with torch.set_grad_enabled(train):
        for x, f, y in loader:
            x, f, y = x.to(device), f.to(device), y.to(device)
            if train and cfg.augment:
                x = augment(x, cfg)
            logits, _ = model(x, f)
            loss = crit(logits, y)
            if train:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            tot_loss += loss.item() * len(y); tot_ok += (logits.argmax(1) == y).sum().item(); n += len(y)
    return tot_loss / n, tot_ok / n


def main():
    ap = argparse.ArgumentParser()
    add_data_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", choices=["brainwaveid", "eegnet"], default="brainwaveid")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no_lstm", action="store_true")
    ap.add_argument("--uniform_weights", action="store_true")
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--hjorth", action="store_true")
    ap.add_argument("--n_channels", type=int, default=32)
    args = ap.parse_args()

    cfg = Config(n_channels=args.n_channels,
             epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                 use_lstm=not args.no_lstm, uniform_band_weights=args.uniform_weights,
                 augment=not args.no_aug, use_hjorth=args.hjorth, train_overlap=args.train_overlap, val_mode=args.val_mode, notch_hz=default_notch(args),
                 artifact_uv=args.artifact_uv)
    set_seed(cfg.seed)
    os.makedirs(args.out, exist_ok=True)
    cfg.save(os.path.join(args.out, "config.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}")

    if args.data == "dreamer":
        prep = prepare_dreamer(args.root, cfg, n_subjects=args.n_subjects or 23, cache=args.cache)
    else:
        prep = prepare(load_recordings(args), cfg, holdout=args.holdout_subjects, cache=args.cache, use_ica=args.ica)
    tr = prep.train.subset(prep.train.cls >= 0)
    va = prep.val.subset(prep.val.cls >= 0)
    if len(prep.trained) < 2 or len(tr.X) == 0 or len(va.X) == 0:
        raise SystemExit(
            "[train] Not enough usable data to train: "
            f"{len(prep.trained)} subject(s) survived artifact rejection with "
            f"{len(tr.X)} train / {len(va.X)} val windows. "
            "Re-run with a higher --artifact_uv (try 150-200), --ica, more subjects "
            "(--n_subjects), or a longer enrollment recording.")

    def mk(s, sh):
        bs = min(cfg.batch_size, len(s.X))                 # never let batch_size exceed dataset size
        return DataLoader(TensorDataset(torch.from_numpy(s.X), torch.from_numpy(s.F),
                                        torch.from_numpy(s.cls).long()),
                          batch_size=bs, shuffle=sh, drop_last=sh and len(s.X) > bs)
    tl, vl = mk(tr, True), mk(va, False)

    model = build_model(args.model, cfg, len(prep.trained), tr.F.shape[1]).to(device)
    print(f"[train] {args.model}: {count_params(model):,} trainable parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best, bad, ckpt = 1e9, 0, os.path.join(args.out, "best.pt")
    for ep in range(1, cfg.epochs + 1):
        t0 = time.time()
        trl, tra = run_epoch(model, tl, crit, device, cfg, opt)
        vll, vla = run_epoch(model, vl, crit, device, cfg)
        sched.step()
        for k, v in zip(hist, (trl, tra, vll, vla)):
            hist[k].append(v)
        print(f"ep {ep:02d}  train {trl:.3f}/{tra:.3f}  val {vll:.3f}/{vla:.3f}  ({time.time() - t0:.1f}s)")
        if vll < best - 1e-4:
            best, bad = vll, 0
            torch.save(dict(model=model.state_dict(), name=args.model, n_classes=len(prep.trained),
                            feat_dim=tr.F.shape[1]), ckpt)
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"early stopping at epoch {ep}"); break
    json.dump(hist, open(os.path.join(args.out, "history.json"), "w"))

    model.load_state_dict(torch.load(ckpt)["model"])
    res = evaluate_run(model, prep, device, args.out, hist)
    res["n_params"] = count_params(model)
    json.dump(res, open(os.path.join(args.out, "results.json"), "w"), indent=2)
    print_summary(res)


if __name__ == "__main__":
    main()
