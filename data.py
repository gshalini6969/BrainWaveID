"""Data loading (synthetic / DEAP / in-house), windowing, subject-wise bookkeeping and splits.

A *recording* is a dict: {raw: (C, N) microvolts, fs, subject, role: 'enroll'|'verify', state: str}.

Split logic (no leakage between train/val/test):
  * role='enroll' recordings -> first (1-val_frac) of the recording = TRAIN, last val_frac = VAL
    (contiguous, and windows straddling the boundary are dropped).
  * role='verify' recordings -> TEST (each with its induced-state label).
"""
import csv, hashlib, json, os, pickle
from dataclasses import dataclass
import numpy as np
from .config import Config
from .signal_proc import (preprocess_recording, make_windows, extract_features,
                          FeatureScaler, n_feats_per_channel)


# ============================================================ loaders
def load_dreamer(root, n_subjects=23):
    """Load processed DREAMERV EEG windows.

    DREAMERV:
      X: (N, 14, 256), already windowed at 128 Hz
      y: binary valence label
      metadata['subject_id']: participant ID 1..23

    Each window is kept as a separate recording because the dataset
    is already segmented into 2-second windows.
    """
    root = os.path.abspath(root)

    x_path = os.path.join(root, "DREAMERV_X.npy")
    y_path = os.path.join(root, "DREAMERV_y.npy")
    m_path = os.path.join(root, "DREAMERV_metadata.npy")

    if not all(os.path.exists(p) for p in (x_path, y_path, m_path)):
        raise FileNotFoundError(
            f"DREAMERV files not found under {root}. "
            "Expected DREAMERV_X.npy, DREAMERV_y.npy and DREAMERV_metadata.npy."
        )

    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path)
    metadata = np.load(m_path, allow_pickle=True).item()
    subject_ids = metadata["subject_id"]

    if X.ndim != 3 or X.shape[1:] != (14, 256):
        raise ValueError(f"Unexpected DREAMERV X shape: {X.shape}")

    if len(X) != len(y) or len(X) != len(subject_ids):
        raise ValueError(
            f"DREAMERV length mismatch: X={len(X)}, y={len(y)}, "
            f"subject_ids={len(subject_ids)}"
        )

    available = sorted(np.unique(subject_ids).tolist())
    selected = available[:n_subjects]

    recs = []

    for sid in selected:
        idx = np.where(subject_ids == sid)[0]

        # Keep both valence classes represented in enrollment and verification.
        for label in (0, 1):
            label_idx = idx[y[idx] == label]

            if len(label_idx) < 2:
                continue

            # 80% enrollment, 20% verification within this subject+valence.
            cut = int(len(label_idx) * 0.8)

            enroll_idx = label_idx[:cut]
            verify_idx = label_idx[cut:]

            subject = f"D{int(sid):03d}"
            state = f"valence_{int(label)}"

            for j in enroll_idx:
                recs.append(dict(
                    raw=X[j],
                    fs=128.0,
                    subject=subject,
                    role="enroll",
                    state=state
                ))

            for j in verify_idx:
                recs.append(dict(
                    raw=X[j],
                    fs=128.0,
                    subject=subject,
                    role="verify",
                    state=state
                ))

    print(
        f"[dreamer] {len(selected)} subjects, "
        f"{len(recs)} windows, "
        f"selected={selected}"
    )

    return recs

def load_deap(root, n_subjects=32, cfg=None):
    """DEAP 'data_preprocessed_python' folder (s01.dat ... s32.dat).
    Uses the 32 EEG channels, drops the 3 s pre-trial baseline. Trials 0-31 = enroll, 32-39 = verify."""
    recs = []
    for s in range(1, n_subjects + 1):
        with open(os.path.join(root, f"s{s:02d}.dat"), "rb") as f:
            d = pickle.load(f, encoding="latin1")
        data = d["data"][:, :32, 384:]                    # (40 trials, 32 ch, 7680) @128 Hz
        for t in range(data.shape[0]):
            recs.append(dict(raw=data[t], fs=128, subject=f"s{s:02d}",
                             role="enroll" if t < 32 else "verify", state="video"))
    return recs


def load_inhouse(manifest):
    """manifest CSV columns: path,subject,role,state,fs
       path -> .npy of shape (channels, samples) in microvolts, or .edf (needs mne).
       role  -> enroll | verify ; state -> baseline / deep_breathing / excited / stressed / drowsy ..."""
    recs, base = [], os.path.dirname(os.path.abspath(manifest))
    with open(manifest) as f:
        for row in csv.DictReader(f):
            p = row["path"] if os.path.isabs(row["path"]) else os.path.join(base, row["path"])
            if p.endswith(".npy"):
                raw = np.load(p)
            elif p.endswith(".edf"):
                import mne
                raw = mne.io.read_raw_edf(p, preload=True, verbose=False).get_data() * 1e6
            else:
                raise ValueError(f"Unsupported file type: {p}")
            recs.append(dict(raw=raw, fs=float(row["fs"]), subject=row["subject"],
                             role=row["role"], state=row["state"]))
    return recs


# ---------------- PhysioNet EEG Motor Movement/Imagery (EEGMMIDB): 109 subjects, 64 ch, 160 Hz
EEGMMIDB_CH = ["Fp1", "Fp2", "Af3", "Af4", "F7", "F3", "Fz", "F4", "F8", "Fc5", "Fc1", "Fc2", "Fc6",
               "T7", "C3", "Cz", "C4", "T8", "Cp5", "Cp1", "Cp2", "Cp6", "P7", "P3", "Pz", "P4", "P8",
               "Po3", "Po4", "O1", "Oz", "O2"]            # 32 of the 64 electrodes (10-20-like subset)
# Common 14-channel montage shared by EEGMMIDB and DREAMER
COMMON14_CH = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]
EEGMMIDB_TASKS = {"fist_exec": [3, 7, 11], "fist_imag": [4, 8, 12],
                  "fists_feet_exec": [5, 9, 13], "fists_feet_imag": [6, 10, 14]}


def _read_eegmmidb(path, t0=None, t1=None):
    import mne
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    look = {n.strip(".").upper(): i for i, n in enumerate(raw.ch_names)}
    picks = [look[c.upper()] for c in EEGMMIDB_CH]
    fs = raw.info["sfreq"]
    x = raw.get_data(picks=picks).astype(np.float32) * 1e6          # volts -> microvolts
    a = 0 if t0 is None else int(t0 * fs)
    b = x.shape[1] if t1 is None else int(t1 * fs)
    return x[:, a:b], fs                                             # (data, sampling rate of THIS file)


def load_eegmmidb(root, protocol="rest_to_task", n_subjects=None, exclude=(), enroll_sec=40):
    """Expects root/S001/S001R01.edf ... (see scripts/download_eegmmidb.py).

    protocol 'rest_to_task'  (mirrors the paper: enroll on SHORT eyes-closed rest only)
        enroll : R02 (eyes closed) first `enroll_sec` s
        verify : baseline = rest of R02 (matched state) | eyes_open = R01 | 4 task states (R03-R14)
    protocol 'run_split'     (standard multi-condition protocol; expect much higher accuracy)
        enroll : R02[:enroll_sec], R01, task runs R03-R10
        verify : baseline = rest of R02 | task runs R11-R14 (repetition 3 of each task type)
    """
    recs, used, skipped = [], 0, []
    for sid in range(1, 110):
        if sid in exclude:
            continue
        if n_subjects and used >= n_subjects:
            break
        f = lambda r: os.path.join(root, f"S{sid:03d}", f"S{sid:03d}R{r:02d}.edf")
        if not all(os.path.exists(f(r)) for r in range(1, 15)):
            skipped.append(sid); continue
        name = f"S{sid:03d}"
        fs = 160.0   # nominal; the true per-file rate returned by the lazy loader takes precedence
        mk = lambda path, t0=None, t1=None: (lambda: _read_eegmmidb(path, t0, t1))
        recs.append(dict(raw=mk(f(2), 0, enroll_sec), fs=fs, subject=name, role="enroll", state="eyes_closed"))
        recs.append(dict(raw=mk(f(2), enroll_sec, None), fs=fs, subject=name, role="verify", state="baseline"))
        if protocol == "rest_to_task":
            recs.append(dict(raw=mk(f(1)), fs=fs, subject=name, role="verify", state="eyes_open"))
        else:
            recs.append(dict(raw=mk(f(1)), fs=fs, subject=name, role="enroll", state="eyes_open"))
        for state, runs in EEGMMIDB_TASKS.items():
            for r in runs:
                role = "verify" if (protocol == "rest_to_task" or r >= 11) else "enroll"
                recs.append(dict(raw=mk(f(r)), fs=fs, subject=name, role=role, state=state))
        used += 1
    if skipped and used:
        print(f"[eegmmidb] skipped {len(skipped)} subjects with missing files (e.g. {skipped[:5]})")
    if not used:
        raise SystemExit(f"[eegmmidb] no complete subjects found under {root}. "
                         f"Run scripts/download_eegmmidb.py first.")
    print(f"[eegmmidb] {used} subjects, {len(recs)} recordings, protocol={protocol}")
    return recs


# ---------------- synthetic generator: ONLY for testing the pipeline, never for reporting results
_STATE = {  # alpha gain, beta gain, noise gain, muscle-burst prob / s
    "enroll": (1.10, 1.0, 1.0, 0.00), "baseline": (1.00, 1.0, 1.0, 0.00),
    "deep_breathing": (1.15, 0.9, 0.9, 0.00), "excited": (0.75, 1.4, 1.3, 0.10),
    "stressed": (0.65, 1.3, 1.2, 0.05),
}


def make_synthetic(n_subjects=8, rec_sec=60, fs=128, n_ch=32, seed=0):
    rng = np.random.default_rng(seed)
    n = rec_sec * fs
    t = np.arange(n) / fs
    post = np.linspace(0.4, 1.4, n_ch)                    # alpha stronger at "posterior" channels
    recs = []

    def osc(f):
        return np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28) + 0.3 * np.cumsum(rng.standard_normal(n)) / np.sqrt(fs))

    for s in range(n_subjects):
        fa, fb, ft = rng.uniform(8.5, 12.5), rng.uniform(15, 26), rng.uniform(4.5, 7)
        M = rng.standard_normal((n_ch, 4))
        M[:, :2] = np.abs(M[:, :2]) * post[:, None]
        sessions = [("enroll", "enroll")] * 2 + [("verify", k) for k in
                                                 ("baseline", "deep_breathing", "excited", "stressed")]
        for role, state in sessions:
            ga, gb, gn, pm = _STATE[state]
            g = np.exp(rng.normal(0, 0.15))
            S = np.stack([osc(fa), osc(fa + 1.5), osc(fb), osc(ft)])
            pink = rng.standard_normal((n_ch, n))
            Fz = np.fft.rfft(pink, axis=-1) / np.sqrt(np.maximum(np.fft.rfftfreq(n, 1 / fs), 0.5))
            pink = np.fft.irfft(Fz, n=n, axis=-1); pink /= pink.std()
            x = (15 * ga * g * (M[:, :2] @ S[:2]) + 8 * gb * (M[:, 2:3] * S[2]) +
                 6 * (M[:, 3:4] * S[3]) + 10 * gn * pink + 1.0 * np.sin(2 * np.pi * 50 * t))
            for sec in range(rec_sec):                    # occasional muscle bursts
                if rng.random() < pm:
                    ch = rng.choice(n_ch, 8, replace=False)
                    x[ch, sec * fs:(sec + 1) * fs] += rng.normal(0, 90, (8, fs))
            recs.append(dict(raw=x, fs=fs, subject=f"syn{s:02d}", role=role, state=state))
    return recs


def load_recordings(args):
    if args.data == "synthetic":
        return make_synthetic(args.n_subjects or 8, seed=args.seed)
    if args.data == "eegmmidb":
        return load_eegmmidb(args.root, args.protocol, args.n_subjects,
                             exclude={int(i) for i in args.exclude.split(",") if i})
    if args.data == "dreamer":
        return load_dreamer(args.root, args.n_subjects or 23)
    if args.data == "deap":
        return load_deap(args.root)
    if args.data == "inhouse":
        return load_inhouse(args.manifest)
    raise ValueError(args.data)


# ============================================================ windows + splits
@dataclass
class Split:
    X: np.ndarray        # (N, C, T) z-scored windows
    F: np.ndarray        # (N, D) standardised + alpha-weighted engineered features
    subj: np.ndarray     # global subject index
    cls: np.ndarray      # class index among TRAINED subjects, -1 for held-out subjects
    state: np.ndarray    # str
    rec: np.ndarray      # recording id

    def subset(self, mask):
        return Split(*(a[mask] for a in (self.X, self.F, self.subj, self.cls, self.state, self.rec)))


@dataclass
class Prepared:
    train: Split
    val: Split
    test: Split
    subjects: list
    trained: list        # global subject indices used to train the network
    scaler: FeatureScaler
    cfg: Config
    reject_stats: dict


def build_windows(recs, cfg: Config, use_ica: bool = False):
    keep_lists = dict(X=[], subj=[], state=[], rec=[], split=[])
    subjects = sorted({r["subject"] for r in recs})
    sidx = {s: i for i, s in enumerate(subjects)}
    tot, kept = {}, {}
    enroll_recs = {}
    for i, r in enumerate(recs):
        if r["role"] == "enroll":
            enroll_recs.setdefault(r["subject"], []).append(i)
    last_enroll = {s: v[-1] for s, v in enroll_recs.items() if len(v) >= 2}   # held-out session per subject
    for i, r in enumerate(recs):
        raw, fs = (r["raw"](), r["fs"]) if callable(r["raw"]) else (r["raw"], r["fs"])   # lazy loaders keep RAM low
        if isinstance(raw, tuple):
            raw, fs = raw
        from .signal_proc import ica_clean
        x = preprocess_recording(raw, fs, cfg, ica_fn=ica_clean if use_ica else None)
        ov = cfg.train_overlap if r["role"] == "enroll" else 0.0
        W, starts = make_windows(x, cfg, ov)
        if len(W) == 0:
            continue
        ok = np.abs(W).max(axis=(1, 2)) <= cfg.artifact_uv      # rejection in uV BEFORE z-scoring
        tot[r["state"]] = tot.get(r["state"], 0) + len(W)
        kept[r["state"]] = kept.get(r["state"], 0) + int(ok.sum())
        mu, sd = x.mean(1), x.std(1) + 1e-8                      # per-channel session z-score
        W = ((W[ok] - mu[None, :, None]) / sd[None, :, None]).astype(np.float32)
        starts = starts[ok]
        if r["role"] == "enroll" and cfg.val_mode == "session" and last_enroll.get(r["subject"]) == i:
            split = np.ones(len(W), dtype=int)                    # whole session -> validation
        elif r["role"] == "enroll" and cfg.val_mode == "session" and r["subject"] in last_enroll:
            split = np.zeros(len(W), dtype=int)                   # other enrollment sessions -> train
        elif r["role"] == "enroll":
            vstart = int(x.shape[1] * (1 - cfg.val_frac))
            split = np.where(starts >= vstart, 1, np.where(starts + cfg.win_len <= vstart, 0, -1))
        else:
            split = np.full(len(W), 2)
        keep_lists["X"].append(W)
        keep_lists["subj"].append(np.full(len(W), sidx[r["subject"]]))
        keep_lists["state"].append(np.full(len(W), r["state"]))
        keep_lists["rec"].append(np.full(len(W), i))
        keep_lists["split"].append(split)
    out = {k: np.concatenate(v) for k, v in keep_lists.items()}
    out["subjects"] = subjects
    out["reject"] = {s: 1 - kept[s] / tot[s] for s in tot}
    return out

def prepare_dreamer(root, cfg: Config, n_subjects=23, cache=None):
    """Prepare DREAMERV directly from its already-windowed NumPy arrays.

    DREAMERV samples are already 2-second, 128-Hz windows with 14 channels.
    Subject identity is the authentication class; valence_0/valence_1 are
    affective conditions.

    Each subject/valence combination is split 80/20 into enrollment/test.
    Enrollment is then split 80/20 into train/validation.
    """
    root = os.path.abspath(root)

    x_path = os.path.join(root, "DREAMERV_X.npy")
    y_path = os.path.join(root, "DREAMERV_y.npy")
    m_path = os.path.join(root, "DREAMERV_metadata.npy")

    Xraw = np.load(x_path, mmap_mode="r")
    y = np.load(y_path)
    metadata = np.load(m_path, allow_pickle=True).item()
    subject_ids = metadata["subject_id"]

    if Xraw.shape[1:] != (14, 256):
        raise ValueError(f"Unexpected DREAMERV shape: {Xraw.shape}")

    selected = sorted(np.unique(subject_ids).tolist())[:n_subjects]

    # ------------------------------------------------------------
    # Split indices subject-wise and valence-wise.
    # ------------------------------------------------------------
    train_idx = []
    val_idx = []
    test_idx = []

    rng = np.random.default_rng(cfg.seed)

    for sid in selected:
        for label in (0, 1):
            idx = np.where((subject_ids == sid) & (y == label))[0]

            if len(idx) < 10:
                continue

            # Deterministic shuffle prevents temporal/trial ordering
            # from becoming the train/test boundary.
            idx = idx.copy()
            rng.shuffle(idx)

            n_test = max(1, int(round(0.20 * len(idx))))
            verify = idx[:n_test]
            enroll = idx[n_test:]

            n_val = max(1, int(round(0.20 * len(enroll))))
            val = enroll[:n_val]
            train = enroll[n_val:]

            train_idx.extend(train.tolist())
            val_idx.extend(val.tolist())
            test_idx.extend(verify.tolist())

    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    # ------------------------------------------------------------
    # Process windows in manageable chunks.
    # ------------------------------------------------------------
    def process(indices):
        if len(indices) == 0:
            return (
                np.empty((0, 14, 256), dtype=np.float32),
                np.empty((0, 168), dtype=np.float32),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=object),
            )

        chunks_x = []
        chunks_f = []
        chunks_s = []
        chunks_state = []

        for start in range(0, len(indices), 2048):
            ii = indices[start:start + 2048]
            W = np.asarray(Xraw[ii], dtype=np.float32)

            # DREAMERV is already at 128 Hz and 2 seconds.
            ok = np.isfinite(W).all(axis=(1, 2))
            W = W[ok]
            ii = ii[ok]

            if len(W) == 0:
                continue

            # Per-window channel z-score, matching the existing
            # session-normalisation philosophy.
            mu = W.mean(axis=2, keepdims=True)
            sd = W.std(axis=2, keepdims=True) + 1e-8
            W = ((W - mu) / sd).astype(np.float32)

            # Artifact rejection after loading, before feature extraction.
            keep = np.abs(W).max(axis=(1, 2)) <= cfg.artifact_uv
            W = W[keep]
            ii = ii[keep]

            if len(W) == 0:
                continue

            F = extract_features(W, cfg.fs_target, cfg)

            chunks_x.append(W)
            chunks_f.append(F)
            chunks_s.append(subject_ids[ii].astype(np.int64))
            chunks_state.append(
                np.where(y[ii] == 0, "valence_0", "valence_1")
            )

        if not chunks_x:
            return (
                np.empty((0, 14, 256), dtype=np.float32),
                np.empty((0, 168), dtype=np.float32),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=object),
            )

        return (
            np.concatenate(chunks_x),
            np.concatenate(chunks_f),
            np.concatenate(chunks_s),
            np.concatenate(chunks_state),
        )

    trX, trF, trS, trState = process(train_idx)
    vaX, vaF, vaS, vaState = process(val_idx)
    teX, teF, teS, teState = process(test_idx)

    subjects = [f"D{int(sid):03d}" for sid in selected]

    # Map raw DREAMER subject IDs to 0..N-1 class IDs.
    sid_to_class = {sid: i for i, sid in enumerate(selected)}

    def convert_subjects(S):
        return np.asarray([sid_to_class[int(s)] for s in S], dtype=np.int64)

    trSub = convert_subjects(trS)
    vaSub = convert_subjects(vaS)
    teSub = convert_subjects(teS)

    # Scale engineered features using TRAIN statistics only.
    scaler = FeatureScaler().fit(trF, cfg, cfg.n_channels)
    trF = scaler.transform(trF)
    vaF = scaler.transform(vaF)
    teF = scaler.transform(teF)

    def make_split(X, F, S, states):
        return Split(
            X=X,
            F=F,
            subj=S,
            cls=S.copy(),
            state=states,
            rec=np.arange(len(X), dtype=np.int64),
        )

    prep = Prepared(
        train=make_split(trX, trF, trSub, trState),
        val=make_split(vaX, vaF, vaSub, vaState),
        test=make_split(teX, teF, teSub, teState),
        subjects=subjects,
        trained=list(range(len(subjects))),
        scaler=scaler,
        cfg=cfg,
        reject_stats={},
    )

    print(
        f"[dreamer] train={len(trX)} val={len(vaX)} test={len(teX)} | "
        f"subjects={len(subjects)} | feat_dim={trF.shape[1]}"
    )

    return prep

def prepare(recs, cfg: Config, holdout: int = 0, cache: str = None, use_ica: bool = False) -> Prepared:
    key = hashlib.md5(json.dumps([cfg.fs_target, cfg.win_sec, cfg.train_overlap, cfg.artifact_uv,
                                  cfg.notch_hz, cfg.bp_low, cfg.bp_high, cfg.val_frac, cfg.val_mode,
                                  use_ica, len(recs)]).encode()).hexdigest()[:10]
    cpath = f"{cache}_{key}.pkl" if cache else None
    if cpath and os.path.exists(cpath):
        with open(cpath, "rb") as f:
            w = pickle.load(f)
        print(f"[data] loaded cached windows from {cpath}")
    else:
        w = build_windows(recs, cfg, use_ica=use_ica)
        if cpath:
            os.makedirs(os.path.dirname(cpath) or ".", exist_ok=True)
            with open(cpath, "wb") as f:
                pickle.dump(w, f, protocol=4)

    m = w["split"] >= 0
    X, subj, state, rec, split = (w[k][m] for k in ("X", "subj", "state", "rec", "split"))
    print(f"[data] {len(X)} windows | rejection rate by state: " +
          ", ".join(f"{k}={v * 100:.1f}%" for k, v in w["reject"].items()))
    print("[data] extracting engineered features ...")
    F = extract_features(X, cfg.fs_target, cfg)

    n_sub = len(w["subjects"])
    rng = np.random.default_rng(cfg.seed)
    held = set(rng.choice(n_sub, holdout, replace=False).tolist()) if holdout else set()

    # Quarantine any subject with too little usable enrollment data (e.g. all windows lost to
    # artifact rejection). Without this a subject can end up with an all-NaN template and crash
    # verification scoring downstream, or a softmax class the network never actually saw examples of.
    min_windows = 3
    train_counts = {i: int(((subj == i) & (split == 0)).sum()) for i in range(n_sub)}
    val_counts = {i: int(((subj == i) & (split == 1)).sum()) for i in range(n_sub)}
    starved = [i for i in range(n_sub) if i not in held and
              (train_counts[i] < min_windows or val_counts[i] < 1)]
    if starved:
        names = [w["subjects"][i] for i in starved]
        print(f"[data] WARNING: {len(starved)} subject(s) have too few enrollment windows after "
              f"artifact rejection (need >={min_windows} train + >=1 val) and are excluded from "
              f"network training: {names}")
        print("[data]   -> raise --artifact_uv (e.g. 150-200), or add --ica, or record a longer "
              "enrollment session for these subjects.")
        held |= set(starved)

    trained = [i for i in range(n_sub) if i not in held]
    cmap = np.full(n_sub, -1); cmap[trained] = np.arange(len(trained))
    cls = cmap[subj]

    tr_fit = (split == 0) & (cls >= 0)
    scaler = FeatureScaler().fit(F[tr_fit], cfg, cfg.n_channels)
    F = scaler.transform(F)

    def mk(s):
        mm = split == s
        return Split(X[mm], F[mm], subj[mm], cls[mm], state[mm], rec[mm])

    prep = Prepared(mk(0), mk(1), mk(2), w["subjects"], trained, scaler, cfg, w["reject"])
    print(f"[data] train={len(prep.train.X)} val={len(prep.val.X)} test={len(prep.test.X)} | "
          f"subjects={n_sub} (trained on {len(trained)}) | feat_dim={F.shape[1]}")
    return prep


def default_notch(args):
    return args.notch_hz if args.notch_hz else (60.0 if args.data == "eegmmidb" else 50.0)


def add_data_args(p):
    p.add_argument("--data", choices=["synthetic", "deap", "inhouse", "eegmmidb", "dreamer"], default="synthetic")
    p.add_argument("--root", help="DEAP data_preprocessed_python folder, or EEGMMIDB root (contains S001/, S002/ ...)")
    p.add_argument("--manifest", help="in-house manifest CSV")
    p.add_argument("--n_subjects", type=int, default=None, help="synthetic (default 8) / eegmmidb (default all 109)")
    p.add_argument("--protocol", choices=["rest_to_task", "run_split"], default="rest_to_task", help="eegmmidb only")
    p.add_argument("--exclude", default="", help="eegmmidb: comma-separated subject numbers to skip, e.g. 88,92")
    p.add_argument("--notch_hz", type=float, default=None, help="mains frequency (default 50; 60 for eegmmidb)")
    p.add_argument("--holdout_subjects", type=int, default=0,
                   help="subjects excluded from network training (open-set verification test)")
    p.add_argument("--train_overlap", type=float, default=0.5)
    p.add_argument("--cache", default=None, help="path prefix for caching preprocessed windows")
    p.add_argument("--val_mode", choices=["tail", "session"], default="tail")
    p.add_argument("--artifact_uv", type=float, default=100.0,
                  help="reject windows with peak amplitude above this (uV). Raise to 150-200 for "
                       "raw/real data without ICA cleaning, which otherwise loses 40-70%% of windows.")
    p.add_argument("--ica", action="store_true",
                  help="apply ICA eye/blink removal before artifact rejection (needs `pip install mne`, "
                       "slower, lets you keep a stricter --artifact_uv)")
    p.add_argument("--seed", type=int, default=42)
    return p
