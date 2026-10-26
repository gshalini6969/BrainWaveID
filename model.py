"""BrainWaveID CNN-LSTM (Sec. III-D, Table III) and an EEGNet baseline."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import Config


class SepBlock(nn.Module):
    """Depthwise-separable Conv1d -> BatchNorm -> ReLU -> MaxPool(2).
    Depthwise: one temporal filter bank per input channel. Pointwise: mixes channels (spatial)."""

    def __init__(self, cin, cout, k):
        super().__init__()
        self.dw = nn.Conv1d(cin, cin, k, padding="same", groups=cin, bias=False)
        self.pw = nn.Conv1d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm1d(cout)          # PyTorch momentum 0.1 == Keras momentum 0.9
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.pw(self.dw(x)))))


class BrainWaveID(nn.Module):
    """forward(x, f) -> (logits, embedding)
    x: (B, 32, 256) raw z-scored window;  f: (B, D) engineered alpha-weighted features."""

    def __init__(self, cfg: Config, n_classes: int, feat_dim: int):
        super().__init__()
        self.cfg = cfg
        self.b1 = SepBlock(cfg.n_channels, cfg.conv1_filters, cfg.conv1_kernel)
        self.b2 = SepBlock(cfg.conv1_filters, cfg.conv2_filters, cfg.conv2_kernel)
        if cfg.use_lstm:
            self.lstm = nn.LSTM(cfg.conv2_filters, cfg.lstm_hidden, cfg.lstm_layers, batch_first=True)
            rep = cfg.lstm_hidden
        else:
            rep = cfg.conv2_filters
        self.fc = nn.Linear(rep + feat_dim, cfg.embed_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.cls = nn.Linear(cfg.embed_dim, n_classes)

    def forward(self, x, f):
        h = self.b2(self.b1(x))                          # (B, 128, T/4)
        if self.cfg.use_lstm:
            _, (hn, _) = self.lstm(h.transpose(1, 2))    # (B, T/4, 128)
            z = hn[-1]
        else:
            z = h.mean(-1)
        e = self.fc(torch.cat([z, f], 1))                # 512-d embedding (used as template)
        return self.cls(self.drop(F.relu(e))), e


class EEGNet(nn.Module):
    """Compact EEGNet-8,2 (Lawhern et al. 2018) baseline; ignores engineered features."""

    def __init__(self, cfg: Config, n_classes: int, F1=8, D=2, F2=16):
        super().__init__()
        C, T = cfg.n_channels, cfg.win_len
        self.net = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding="same", bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(0.25),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding="same", groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(0.25), nn.Flatten())
        with torch.no_grad():
            d = self.net(torch.zeros(2, 1, C, T)).shape[1]
        self.cls = nn.Linear(d, n_classes)

    def forward(self, x, f=None):
        e = self.net(x.unsqueeze(1))
        return self.cls(e), e


def build_model(name, cfg, n_classes, feat_dim):
    if name == "brainwaveid":
        return BrainWaveID(cfg, n_classes, feat_dim)
    if name == "eegnet":
        return EEGNet(cfg, n_classes)
    raise ValueError(name)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
