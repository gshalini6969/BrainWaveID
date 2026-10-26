"""Central configuration for BrainWaveID. Every hyper-parameter from the paper lives here."""
from dataclasses import dataclass, asdict
import json

# Frequency bands (Hz) - Table VII of the paper
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass
class Config:
    # ---------------- signal / preprocessing ----------------
    # NOTE: 2 s x 128 Hz = 256 samples, which matches the 32x256 input in Table III.
    # (2 s at 256 Hz would be 512 samples.) DEAP's preprocessed data is 128 Hz as well.
    fs_target: int = 128
    n_channels: int = 32
    win_sec: float = 2.0
    train_overlap: float = 0.5      # overlap for ENROLLMENT windows only (augments small data)
    artifact_uv: float = 100.0      # reject windows with |amplitude| > 100 uV
    notch_hz: float = 50.0          # India mains = 50 Hz
    bp_low: float = 0.5
    bp_high: float = 45.0
    val_frac: float = 0.2           # tail of each enrollment recording used as validation
    # "tail"   : validation = last val_frac of each enrollment recording (same session as training)
    # "session": validation = the LAST enrollment recording of each subject, held out entirely
    #            (cross-session early stopping + threshold calibration; recommended for real data)
    val_mode: str = "tail"

    # ---------------- features ----------------
    use_hjorth: bool = False        # paper lists Hjorth but counts only 12 feats/channel
    uniform_band_weights: bool = False   # ablation: "uniform band weighting"
    w_alpha: float = 1.0            # High
    w_beta: float = 0.6             # Medium
    w_other: float = 0.3            # Low (delta, theta, gamma)

    # ---------------- model ----------------
    conv1_filters: int = 64
    conv1_kernel: int = 32
    conv2_filters: int = 128
    conv2_kernel: int = 16
    lstm_hidden: int = 128
    lstm_layers: int = 2
    embed_dim: int = 512
    dropout: float = 0.5
    use_lstm: bool = True           # ablation: "CNN only"

    # ---------------- training ----------------
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 50
    patience: int = 8
    label_smoothing: float = 0.05
    augment: bool = True
    aug_snr_db: float = 20.0
    aug_max_shift_ms: float = 100.0
    aug_channel_drop: float = 0.1
    seed: int = 42

    @property
    def win_len(self) -> int:
        return int(round(self.win_sec * self.fs_target))

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path):
        with open(path) as f:
            return Config(**json.load(f))
