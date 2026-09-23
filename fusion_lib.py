"""Vendored text-fusion block of zen-image-edit: pure torch, no diffusers.

Copied from `transformer.py` of https://huggingface.co/AiArtLab/zen-image-edit so this ComfyUI node
does not need that repo on `sys.path` (nor diffusers installed). Keep in sync with `transformer.py`:
same classes, same `text_fusion_config` keys.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PerLayerNorm(nn.Module):
    """Normalize every slice separately, then the concatenation.

    Without per-slice normalization the early layers barely reach the output: hidden-state norms
    grow roughly 40x from layer 2 to layer 27.
    """

    def __init__(self, in_dim, n_slices):
        super().__init__()
        assert in_dim % n_slices == 0, f"{in_dim} is not divisible by {n_slices}"
        self.n, self.d = n_slices, in_dim // n_slices
        self.per = nn.LayerNorm(self.d, elementwise_affine=False)
        self.all = nn.LayerNorm(in_dim)

    def forward(self, x):
        b, l, _ = x.shape
        return self.all(self.per(x.view(b, l, self.n, self.d)).reshape(b, l, -1))


class AttnBlock(nn.Module):
    """Pre-norm self-attention + FFN. No causality: text is not autoregressive and the DiT
    already sees the whole sequence at once."""

    def __init__(self, d_model, n_heads, ffn_mult=4):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.n2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * ffn_mult),
                                 nn.GELU(approximate="tanh"),
                                 nn.Linear(d_model * ffn_mult, d_model))

    def forward(self, x, key_padding_mask=None):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False, key_padding_mask=key_padding_mask)[0]
        return x + self.ffn(self.n2(x))


class AttnMixer(nn.Module):
    """Token branch: norm -> project to d_model -> N blocks -> project to out_dim.

    The last projection is zero-initialized, so at step 0 the branch adds nothing.
    """

    def __init__(self, in_dim, out_dim, d_model=1024, n_heads=8, blocks=2, max_len=256):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.inp = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.blocks = nn.ModuleList([AttnBlock(d_model, n_heads) for _ in range(blocks)])
        self.out = nn.Linear(d_model, out_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, key_padding_mask=None):
        h = self.inp(self.norm(x))
        n = h.shape[1]
        pos = self.pos
        if n > pos.shape[1]:
            # The position table was trained on `max_len` slots; the tail is padded with zeros
            # (long sequences simply carry no positional signal there, but nothing crashes).
            pos = F.pad(pos, (0, 0, 0, n - pos.shape[1]))
        h = h + pos[:, :n]
        for block in self.blocks:
            h = block(h, key_padding_mask)
        return self.out(h)


class SliceMixer(nn.Module):
    """Attention over the slice axis (the `layerwise_blocks` + `projector` part of Krea 2 fusion).

    Slices are normalized one by one and run as a sequence of length K through N self-attention
    blocks; `Linear(K -> 1)` then collapses the axis. The result is added to the MLP output
    (zero-initialized output projection).
    """

    def __init__(self, n_slices, d, out_dim, n_heads=8, blocks=2, ffn_mult=2):
        super().__init__()
        self.n, self.d = n_slices, d
        self.per = nn.LayerNorm(d, elementwise_affine=False)
        self.blocks = nn.ModuleList([AttnBlock(d, n_heads, ffn_mult) for _ in range(blocks)])
        self.proj = nn.Linear(n_slices, 1, bias=False)
        self.out = nn.Linear(d, out_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        b, l, _ = x.shape
        h = self.per(x.view(b, l, self.n, self.d)).reshape(b * l, self.n, self.d)
        for block in self.blocks:
            h = block(h)
        h = self.proj(h.permute(0, 2, 1)).squeeze(-1)
        return self.out(h).reshape(b, l, -1)


class AttnAdapter(nn.Module):
    """Point-wise MLP (`self.mlp`) plus residual branches over tokens (`self.attn`) and slices (`self.mixer`)."""

    def __init__(self, mods, attn=None, mixer=None):
        super().__init__()
        self.mlp = nn.Sequential(*mods)
        self.attn = attn
        self.mixer = mixer

    def forward(self, x, mask=None):
        """`mask`: `(B, L)` bool, True = real token. The attention branches get `key_padding_mask = ~mask`,
        otherwise attention would look into the padding."""
        out = self.mlp(x)
        if self.attn is not None:
            kpm = None if mask is None else ~mask.bool()
            out = out + self.attn(x, kpm)
        if self.mixer is not None:
            out = out + self.mixer(x)
        return out


def build_fusion(in_dim, out_dim, hidden=4096, proj_layers=2, norm="none", n_slices=1,
                 attention=0, attn_dim=1024, attn_heads=8, max_len=256,
                 mixer=0, mixer_heads=8, mixer_ffn=2, **_ignored):
    """Build the fusion block from the transformer config (`text_fusion_config`).

    `proj_layers` linear layers with GELU(tanh) between them; `attention`/`mixer` are the number of
    residual branches. Extra config keys (`student_layers`, `drop_idx`) are ignored here: the
    pipeline uses them, the block does not.
    """
    if proj_layers < 2:
        raise ValueError("at least 2 linear layers are required")
    mods = []
    if norm == "ln":
        mods.append(nn.LayerNorm(in_dim))
    elif norm == "ln-per-layer":
        mods.append(PerLayerNorm(in_dim, n_slices))
    elif norm == "rms":
        mods.append(nn.RMSNorm(in_dim))
    elif norm != "none":
        raise ValueError(f"unknown norm: {norm}")
    mods += [nn.Linear(in_dim, hidden), nn.GELU(approximate="tanh")]
    for _ in range(proj_layers - 2):
        mods += [nn.Linear(hidden, hidden), nn.GELU(approximate="tanh")]
    mods.append(nn.Linear(hidden, out_dim))
    attn = AttnMixer(in_dim, out_dim, attn_dim, attn_heads, attention, max_len) if attention else None
    mix = SliceMixer(n_slices, in_dim // n_slices, out_dim, mixer_heads, mixer, mixer_ffn) if mixer else None
    if attn is not None or mix is not None:
        return AttnAdapter(mods, attn, mix)
    return nn.Sequential(*mods)
