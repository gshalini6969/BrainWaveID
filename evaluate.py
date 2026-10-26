"""Evaluation: closed-set identification + verification (FAR / FRR / EER), per mental state."""
import json, os
import numpy as np
import torch
from .data import Prepared
from sklearn.metrics import roc_curve, precision_recall_fscore_support, confusion_matrix


# ---------------------------------------------------------------- helpers
@torch.no_grad()
def embed(model, split, device, bs=512):
    model.eval()
    L, E = [], []
    for i in range(0, len(split.X), bs):
        x = torch.from_numpy(split.X[i:i + bs]).to(device)
        f = torch.from_numpy(split.F[i:i + bs]).to(device)
        lo, e = model(x, f)
        L.append(lo.cpu().numpy()); E.append(e.cpu().numpy())
    return np.concatenate(L), np.concatenate(E)


def l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def eer_curve(gen, imp):
    y = np.r_[np.ones(len(gen)), np.zeros(len(imp))]
    s = np.r_[gen, imp]
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    d = np.abs(fnr - fpr)
    ties = np.where(d <= d.min() + 1e-9)[0]            # many thresholds tie when scores are separable
    ties = ties[np.isfinite(thr[ties])] if np.isfinite(thr[ties]).any() else ties
    i = int(ties[len(ties) // 2])                        # take the middle one, not the strictest
    return dict(eer=float((fpr[i] + fnr[i]) / 2), thr=float(thr[i]), fpr=fpr, tpr=tpr)


def far_frr(gen, imp, tau):
    return float((imp >= tau).mean()), float((gen < tau).mean())


def split_scores(S, subj):
    idx = np.arange(len(subj))
    gen = S[idx, subj]
    mask = np.ones_like(S, bool); mask[idx, subj] = False
    return gen, S[mask]


def make_templates(E, subj, n_sub, mu):
    """Returns (templates, valid_mask). valid_mask[s]=False means subject s had zero enrollment
    windows in the train split (e.g. lost to artifact rejection) - those subjects are excluded
    from scoring rather than producing a NaN template that would poison every similarity score."""
    T = np.zeros((n_sub, E.shape[1]))
    En = l2n(E - mu)
    valid = np.zeros(n_sub, dtype=bool)
    empty = []
    for s in range(n_sub):
        m = subj == s
        if m.any():
            T[s] = En[m].mean(0); valid[s] = True
        else:
            empty.append(s)
    if empty:
        print(f"[eval] WARNING: {len(empty)} subject(s) had no enrollment windows in the train "
              f"split and are excluded from verification scoring: indices {empty}")
    return l2n(np.where(valid[:, None], T, 0.0)), valid


# ---------------------------------------------------------------- main evaluation
def evaluate_run(model, prep, device, out_dir, history=None):
    os.makedirs(out_dir, exist_ok=True)
    n_sub = len(prep.subjects)
    res = {}

    # ---- templates from ENROLLMENT (train-portion) windows of ALL subjects (incl. held-out ones)
    _, E_tr = embed(model, prep.train, device)
    mu = E_tr.mean(0)                                         # population centring (no probe labels used)
    T, valid = make_templates(E_tr, prep.train.subj, n_sub, mu)

    # Drop test/val windows belonging to subjects with no usable template - they'd otherwise
    # score against an all-zero row and silently corrupt EER/FAR/FRR.
    keep_va = valid[prep.val.subj]
    keep_te = valid[prep.test.subj]
    if (~keep_va).any() or (~keep_te).any():
        prep = Prepared(prep.train, prep.val.subset(keep_va), prep.test.subset(keep_te),
                        prep.subjects, prep.trained, prep.scaler, prep.cfg, prep.reject_stats)

    # ---- threshold calibrated on VALIDATION windows (enrollment-session tail) at the EER point
    _, E_va = embed(model, prep.val, device)
    S_va = l2n(E_va - mu) @ T.T
    g, i_ = split_scores(S_va, prep.val.subj)
    tau = eer_curve(g, i_)["thr"]
    res["threshold_tau"] = tau

    # ---- test set
    logits, E_te = embed(model, prep.test, device)
    S_te = l2n(E_te - mu) @ T.T
    st = prep.test.state
    states = sorted(set(st))

    # closed-set identification (only subjects the network was trained on)
    tm = prep.test.cls >= 0
    if tm.any():
        y, p = prep.test.cls[tm], logits[tm].argmax(1)
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
        res["identification"] = dict(accuracy=float((y == p).mean()), precision=float(pr),
                                     recall=float(rc), f1=float(f1), n_windows=int(tm.sum()))
        res["identification"]["per_state_accuracy"] = {
            s: float((y[st[tm] == s] == p[st[tm] == s]).mean()) for s in states if (st[tm] == s).any()}
        cm = confusion_matrix(y, p, normalize="true")
    else:
        cm = None
    # template-matching identification (works for held-out subjects too)
    res["template_id_accuracy"] = float((S_te.argmax(1) == prep.test.subj).mean())

    # verification: overall + per state (tau fixed from validation; per-state EER with own optimum)
    gen, imp = split_scores(S_te, prep.test.subj)
    ov = eer_curve(gen, imp)
    far, frr = far_frr(gen, imp, tau)
    res["verification_overall"] = dict(EER=ov["eer"], FAR_at_tau=far, FRR_at_tau=frr)
    res["verification_per_state"] = {}
    for s in states:
        m = st == s
        g_s, i_s = split_scores(S_te[m], prep.test.subj[m])
        e = eer_curve(g_s, i_s)
        fa, fr = far_frr(g_s, i_s, tau)
        res["verification_per_state"][s] = dict(EER=e["eer"], FAR_at_tau=fa, FRR_at_tau=fr)

    # state-mismatch drop:  A_matched - mean(A_mismatched)   (Sec. III-E)
    pa = res.get("identification", {}).get("per_state_accuracy", {})
    if "baseline" in pa and len(pa) > 1:
        others = [v for k, v in pa.items() if k != "baseline"]
        res["state_mismatch_drop"] = float(pa["baseline"] - np.mean(others))

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    _plots(res, ov, cm, history, out_dir)
    return res


def _plots(res, ov, cm, history, out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if history:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].plot(history["train_acc"], label="train"); ax[0].plot(history["val_acc"], "--", label="validation")
        ax[0].set(title="Accuracy", xlabel="epoch"); ax[0].legend()
        ax[1].plot(history["train_loss"], label="train"); ax[1].plot(history["val_loss"], "--", label="validation")
        ax[1].set(title="Loss", xlabel="epoch"); ax[1].legend()
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.plot(ov["fpr"], ov["tpr"]); ax.plot([0, 1], [0, 1], "k:")
    ax.set(xlabel="False Acceptance Rate", ylabel="Genuine Acceptance Rate",
           title=f"ROC (EER = {ov['eer'] * 100:.2f}%)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "roc.png"), dpi=200); plt.close(fig)
    if cm is not None:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1); fig.colorbar(im)
        ax.set(xlabel="Predicted user", ylabel="True user", title="Confusion matrix (identification)")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200); plt.close(fig)
    pa = res.get("identification", {}).get("per_state_accuracy")
    if pa:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.bar(list(pa), [v * 100 for v in pa.values()])
        ax.set(ylabel="Accuracy (%)", title="Accuracy by verification state (enrolled in enrollment protocol)", ylim=(0, 100))
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "state_accuracy.png"), dpi=200); plt.close(fig)


def print_summary(res):
    idn = res.get("identification")
    if idn:
        print(f"\nIDENTIFICATION  acc={idn['accuracy']*100:.2f}  P={idn['precision']*100:.2f}  "
              f"R={idn['recall']*100:.2f}  F1={idn['f1']*100:.2f}")
        for s, a in idn["per_state_accuracy"].items():
            print(f"   {s:<16s} acc={a*100:.2f}")
    print(f"TEMPLATE-ID acc={res['template_id_accuracy']*100:.2f}")
    v = res["verification_overall"]
    print(f"VERIFICATION    EER={v['EER']*100:.2f}  FAR@tau={v['FAR_at_tau']*100:.2f}  FRR@tau={v['FRR_at_tau']*100:.2f}")
    for s, d in res["verification_per_state"].items():
        print(f"   {s:<16s} EER={d['EER']*100:.2f}  FAR={d['FAR_at_tau']*100:.2f}  FRR={d['FRR_at_tau']*100:.2f}")
    if "state_mismatch_drop" in res:
        print(f"State-mismatch drop = {res['state_mismatch_drop']*100:.2f} pts")
