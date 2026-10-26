"""Preprocessing (Sec. III-B) and feature extraction (Sec. III-C)."""
from math import gcd
import numpy as np
from scipy import signal
from .config import BANDS, Config


# ------------------------------------------------------------------ preprocessing
def preprocess_recording(raw: np.ndarray, fs: float, cfg: Config, ica_fn=None) -> np.ndarray:
    """raw: (channels, samples) in microvolts. Returns filtered, re-referenced, resampled signal.

    Order: [ICA] -> 0.5-45 Hz Butterworth (order 4, zero-phase) -> 50 Hz notch
           -> average re-reference -> resample to cfg.fs_target.
    """
    x = np.asarray(raw, dtype=np.float64)
    if ica_fn is not None:
        x = ica_fn(x, fs)
    nyq = fs / 2.0
    hi = min(cfg.bp_high, 0.99 * nyq)
    sos = signal.butter(4, [cfg.bp_low, hi], btype="bandpass", fs=fs, output="sos")
    x = signal.sosfiltfilt(sos, x, axis=-1)
    if cfg.notch_hz and cfg.notch_hz < nyq:
        b, a = signal.iirnotch(cfg.notch_hz, Q=30.0, fs=fs)
        x = signal.filtfilt(b, a, x, axis=-1)
    x = x - x.mean(axis=0, keepdims=True)                      # average reference
    if int(round(fs)) != cfg.fs_target:
        g = gcd(int(round(fs)), cfg.fs_target)
        x = signal.resample_poly(x, cfg.fs_target // g, int(round(fs)) // g, axis=-1)
    return x


def make_windows(x: np.ndarray, cfg: Config, overlap: float = 0.0):
    """Slice (C, N) into (n, C, win_len) windows. Returns windows and start indices."""
    C, N = x.shape
    w = cfg.win_len
    hop = max(1, int(round(w * (1.0 - overlap))))
    starts = np.arange(0, N - w + 1, hop)
    if len(starts) == 0:
        return np.empty((0, C, w)), starts
    return np.stack([x[:, s:s + w] for s in starts]), starts


def ica_clean(x: np.ndarray, fs: float, ch_names=None, eog_proxy=("Fp1", "Fp2")):
    """OPTIONAL ICA artifact removal via MNE (pip install mne). UNTESTED on real data -
    inspect the rejected components visually for at least a few recordings.
    Rejects components correlated with frontal channels (blink proxy), as in the paper."""
    import mne
    n_ch = x.shape[0]
    ch_names = ch_names or [f"EEG{i:02d}" for i in range(n_ch)]
    info = mne.create_info(ch_names, fs, ch_types="eeg")
    raw = mne.io.RawArray(x * 1e-6, info, verbose=False)
    fit_raw = raw.copy().filter(1.0, None, verbose=False)
    ica = mne.preprocessing.ICA(n_components=min(20, n_ch - 1), method="fastica",
                                random_state=0, max_iter="auto")
    ica.fit(fit_raw, verbose=False)
    bad = set()
    for ch in eog_proxy:
        if ch in ch_names:
            idx, _ = ica.find_bads_eog(raw, ch_name=ch, verbose=False)
            bad.update(idx)
    ica.exclude = sorted(bad)
    ica.apply(raw, verbose=False)
    return raw.get_data() * 1e6


# ------------------------------------------------------------------ features
def _band_power(psd, freqs, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    return psd[..., m].sum(-1) * (freqs[1] - freqs[0])


def _hjorth(W):
    d1 = np.diff(W, axis=-1)
    d2 = np.diff(d1, axis=-1)
    v0, v1, v2 = W.var(-1) + 1e-12, d1.var(-1) + 1e-12, d2.var(-1) + 1e-12
    mob = np.sqrt(v1 / v0)
    comp = np.sqrt(v2 / v1) / mob
    return np.stack([np.log(v0), mob, comp], -1)


def n_feats_per_channel(cfg: Config) -> int:
    return 5 + 5 + 2 + (3 if cfg.use_hjorth else 0)


def extract_features(W: np.ndarray, fs: int, cfg: Config, chunk: int = 1024) -> np.ndarray:
    """W: (N, C, T). Returns (N, C * n_feats) float32.
    Per channel: 5 log-PSD band powers, 5 differential-entropy values, 2 band ratios (+3 Hjorth)."""
    if len(W) == 0:
        return np.empty((0, W.shape[1] * n_feats_per_channel(cfg)), dtype=np.float32)
    out = []
    band_sos = [signal.butter(4, [lo, min(hi, 0.99 * fs / 2)], btype="bandpass", fs=fs, output="sos")
                for lo, hi in BANDS.values()]
    for i in range(0, len(W), chunk):
        w = W[i:i + chunk].astype(np.float64)
        freqs, psd = signal.welch(w, fs=fs, nperseg=min(w.shape[-1], fs), noverlap=min(w.shape[-1], fs) // 2, axis=-1)
        P = np.stack([_band_power(psd, freqs, lo, hi) for lo, hi in BANDS.values()], -1)   # (n,C,5)
        logP = np.log10(P + 1e-12)
        de = np.stack([0.5 * np.log(2 * np.pi * np.e * (signal.sosfiltfilt(s, w, axis=-1).var(-1) + 1e-12))
                       for s in band_sos], -1)                                              # (n,C,5)
        ratios = np.stack([logP[..., 1] - logP[..., 2],       # log(theta/alpha)
                           logP[..., 2] - logP[..., 3]], -1)   # log(alpha/beta)
        parts = [logP, de, ratios] + ([_hjorth(w)] if cfg.use_hjorth else [])
        f = np.concatenate(parts, -1)
        out.append(f.reshape(len(w), -1).astype(np.float32))
    return np.concatenate(out, 0)


def band_weight_vector(cfg: Config) -> np.ndarray:
    """Per-feature weights implementing the High/Medium/Low column of Table VII."""
    if cfg.uniform_band_weights:
        bw = np.ones(5)
    else:
        bw = np.array([cfg.w_other, cfg.w_other, cfg.w_alpha, cfg.w_beta, cfg.w_other])
    w = list(bw) + list(bw) + [bw[2], bw[2]]                   # PSD, DE, ratios (alpha-related)
    if cfg.use_hjorth:
        w += [1.0 if cfg.uniform_band_weights else cfg.w_other] * 3
    return np.asarray(w, dtype=np.float32)


class FeatureScaler:
    """Standardise each feature with TRAIN statistics, then apply alpha-emphasis weights."""

    def fit(self, F: np.ndarray, cfg: Config, n_channels: int):
        if len(F) == 0:
            raise ValueError("FeatureScaler.fit got zero training windows - no subject survived "
                             "artifact rejection with enough enrollment data. Raise --artifact_uv, "
                             "add --ica, or use more/longer recordings.")
        self.mu = F.mean(0)
        self.sd = F.std(0) + 1e-6
        self.w = np.tile(band_weight_vector(cfg), n_channels)
        return self

    def transform(self, F: np.ndarray) -> np.ndarray:
        return (((F - self.mu) / self.sd) * self.w).astype(np.float32)
