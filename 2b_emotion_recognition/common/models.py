"""Model blocks for the three pipelines.

Functional blocks (assignment):  preprocessing -> feature extraction -> temporal/contextual modelling -> fusion -> classifier

  speech : log-mel (T x 80)  -> 1-D CNN front-end -> 2-layer BiGRU -> attention pooling        (temporal block)
  text   : frozen DistilBERT token states (L x 768) -> projection -> BiGRU -> attention pooling (contextual block)
  fusion : speech vector + text vector -> {concat | gated | cross-attention} -> fusion vector    (fusion block)
  head   : MLP classifier over 7 emotions                                                         (classifier block)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

N_CLASSES = 7


class AttnPool(nn.Module):
    """Attention pooling over the valid time steps: sum_t softmax(v^T tanh(W h_t)) h_t."""

    def __init__(self, dim):
        super().__init__()
        self.w = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, h, mask):                                    # h (B,T,D), mask (B,T) bool
        s = self.v(torch.tanh(self.w(h))).squeeze(-1).masked_fill(~mask, -1e9)
        a = torch.softmax(s, dim=1)
        return (a.unsqueeze(-1) * h).sum(1), a


class MeanPool(nn.Module):
    """Masked mean over time (ablation of the attention pooling)."""

    def forward(self, h, mask):
        m = mask.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1), None


def run_gru(gru, x, mask):
    """BiGRU over padded batches, ignoring the padding (packed sequences)."""
    lengths = mask.sum(1).clamp(min=1).cpu()
    packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    out, _ = gru(packed)
    out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
    return out


class SpeechEncoder(nn.Module):
    """arch: "full" = CNN + BiGRU (default) | "cnn" = CNN only | "gru" = BiGRU only (ablations);  pool: "attn" | "mean"."""

    def __init__(self, n_mels=80, ch=128, hidden=128, dropout=0.3, arch="full", pool="attn"):
        super().__init__()
        self.arch = arch
        if arch in ("full", "cnn"):
            self.conv = nn.Sequential(
                nn.Conv1d(n_mels, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(ch, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.ReLU(), nn.MaxPool1d(2),
            )
        else:
            self.conv = None
            self.inproj = nn.Sequential(nn.Linear(n_mels, ch), nn.ReLU(), nn.Dropout(dropout))
        if arch in ("full", "gru"):
            self.gru = nn.GRU(ch, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        else:
            self.gru = None
            self.outproj = nn.Linear(ch, 2 * hidden)
        self.pool = AttnPool(2 * hidden) if pool == "attn" else MeanPool()
        self.out_dim = 2 * hidden

    def forward(self, x, mask):                                    # x (B,T,n_mels) standardised features
        if self.conv is not None:
            h = self.conv(x.transpose(1, 2)).transpose(1, 2)        # (B,T/2,ch)
            m = F.max_pool1d(mask.float().unsqueeze(1), 2).squeeze(1) > 0
        else:
            h, m = self.inproj(x), mask
        seq = run_gru(self.gru, h, m) if self.gru is not None else self.outproj(h)
        vec, _ = self.pool(seq, m)
        return seq, m, vec


class TextEncoder(nn.Module):
    def __init__(self, in_dim=768, hidden=128, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(dropout))
        self.gru = nn.GRU(256, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.pool = AttnPool(2 * hidden)
        self.out_dim = 2 * hidden

    def forward(self, x, mask):                                    # x (B,L,768) frozen DistilBERT states
        seq = run_gru(self.gru, self.proj(x), mask)
        vec, _ = self.pool(seq, mask)
        return seq, mask, vec


class Head(nn.Module):
    def __init__(self, dim, n_classes=N_CLASSES, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, v):
        return self.net(v)


class SpeechModel(nn.Module):
    modality = "speech"

    def __init__(self, cfg=None, **kw):
        super().__init__()
        self.enc = SpeechEncoder(**(cfg or {}))
        self.head = Head(self.enc.out_dim)

    def forward(self, batch, return_reps=False):
        _, _, v = self.enc(batch["speech"], batch["speech_mask"])
        logits = self.head(v)
        return (logits, {"temporal": v}) if return_reps else logits


class TextModel(nn.Module):
    modality = "text"

    def __init__(self, **kw):
        super().__init__()
        self.enc = TextEncoder()
        self.head = Head(self.enc.out_dim)

    def forward(self, batch, return_reps=False):
        _, _, v = self.enc(batch["text"], batch["text_mask"])
        logits = self.head(v)
        return (logits, {"contextual": v}) if return_reps else logits


class FusionModel(nn.Module):
    modality = "fusion"

    def __init__(self, mode="concat", dropout=0.3, cfg=None, **kw):
        super().__init__()
        self.mode = mode
        cfg = dict(cfg or {})
        self.moddrop = float(cfg.pop("moddrop", 0.0))   # modality dropout (extras): probability of hiding the speech / the text during training
        self.speech, self.text = SpeechEncoder(**cfg), TextEncoder()
        d = self.speech.out_dim
        if mode == "concat":
            self.fuse = nn.Sequential(nn.Linear(d + self.text.out_dim, d), nn.ReLU(), nn.Dropout(dropout))
        elif mode == "gated":
            self.gate = nn.Linear(d + self.text.out_dim, d)
            self.tproj = nn.Linear(self.text.out_dim, d)
            self.drop = nn.Dropout(dropout)
        elif mode == "cross":                                       # speech frames attend to the text tokens
            self.tproj = nn.Linear(self.text.out_dim, d)
            self.attn = nn.MultiheadAttention(d, 4, dropout=0.1, batch_first=True)
            self.norm = nn.LayerNorm(d)
            self.pool = AttnPool(d)
        else:
            raise ValueError(mode)
        self.head = Head(d)

    def forward(self, batch, return_reps=False):
        s_seq, s_mask, s_vec = self.speech(batch["speech"], batch["speech_mask"])
        t_seq, t_mask, t_vec = self.text(batch["text"], batch["text_mask"])
        if self.training and self.moddrop > 0:         # each sample: speech hidden with prob p, text hidden with prob p (never both)
            r = torch.rand(s_vec.shape[0], device=s_vec.device)
            ks = (r >= self.moddrop).float().unsqueeze(1)
            kt = ((r < self.moddrop) | (r >= 2 * self.moddrop)).float().unsqueeze(1)
            s_vec, t_vec, s_seq, t_seq = s_vec * ks, t_vec * kt, s_seq * ks.unsqueeze(2), t_seq * kt.unsqueeze(2)
        if self.mode == "concat":
            f = self.fuse(torch.cat([s_vec, t_vec], dim=1))
        elif self.mode == "gated":
            g = torch.sigmoid(self.gate(torch.cat([s_vec, t_vec], dim=1)))
            f = self.drop(g * s_vec + (1 - g) * torch.tanh(self.tproj(t_vec)))
        else:
            kv = self.tproj(t_seq)
            att, _ = self.attn(s_seq, kv, kv, key_padding_mask=~t_mask)
            f, _ = self.pool(self.norm(s_seq + att), s_mask)
        logits = self.head(f)
        if return_reps:
            return logits, {"temporal": s_vec, "contextual": t_vec, "fusion": f}
        return logits


def build_model(modality, fusion_mode="concat", cfg=None):
    """cfg: optional SpeechEncoder options {"n_mels": input dim, "arch": ..., "pool": ...} (extras; the defaults are the main model)."""
    if modality == "speech":
        return SpeechModel(cfg=cfg)
    if modality == "text":
        return TextModel()
    if modality == "fusion":
        return FusionModel(mode=fusion_mode, cfg=cfg)
    raise ValueError(modality)
