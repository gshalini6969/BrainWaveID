"""Classical baselines on the SAME engineered features: SVM (RBF) and KNN (k=5).
   python -m brainwaveid.baselines --data synthetic --out runs/baselines
(EEGNet baseline:  python -m brainwaveid.train --model eegnet ...)"""
import argparse, json, os
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
from .config import Config
from .data import add_data_args, default_notch, load_recordings, prepare
from .evaluate import eer_curve


def main():
    ap = argparse.ArgumentParser(); add_data_args(ap); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = Config(seed=args.seed, train_overlap=args.train_overlap, val_mode=args.val_mode, notch_hz=default_notch(args),
                 artifact_uv=args.artifact_uv)
    prep = prepare(load_recordings(args), cfg, holdout=0, cache=args.cache, use_ica=args.ica)
    tr, te = prep.train, prep.test
    if len(prep.trained) < 2 or len(tr.X) == 0 or len(te.X) == 0:
        raise SystemExit(
            "[baselines] Not enough usable data: "
            f"{len(prep.trained)} subject(s) survived artifact rejection with "
            f"{len(tr.X)} train / {len(te.X)} test windows. "
            "Re-run with a higher --artifact_uv (try 150-200), --ica, more subjects "
            "(--n_subjects), or a longer enrollment recording.")
    out = {}
    for name, clf in [("SVM_RBF", SVC(kernel="rbf", probability=True, random_state=cfg.seed)),
                      ("KNN_k5", KNeighborsClassifier(5))]:
        clf.fit(tr.F, tr.cls)
        proba = clf.predict_proba(te.F); pred = proba.argmax(1)
        pr, rc, f1, _ = precision_recall_fscore_support(te.cls, pred, average="macro", zero_division=0)
        idx = np.arange(len(te.cls)); gen = proba[idx, te.cls]
        m = np.ones_like(proba, bool); m[idx, te.cls] = False
        r = dict(accuracy=float((pred == te.cls).mean()), precision=float(pr), recall=float(rc),
                 f1=float(f1), EER=eer_curve(gen, proba[m])["eer"], per_state_accuracy={
                     s: float((pred[te.state == s] == te.cls[te.state == s]).mean()) for s in sorted(set(te.state))})
        out[name] = r
        print(name, json.dumps({k: (round(v, 4) if not isinstance(v, dict) else {a: round(b, 4) for a, b in v.items()})
                                for k, v in r.items()}))
    os.makedirs(args.out, exist_ok=True)
    json.dump(out, open(os.path.join(args.out, "baselines.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
