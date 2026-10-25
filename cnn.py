"""
The chromosome (Genome) and the CNN it decodes into.

Genes
    n_blocks   1..5 convolution blocks
    filters    5 filter counts, one per possible block, each from {8, 16, 32, 64, 128}
               (genes for blocks beyond n_blocks are carried along but unused, like introns,
               so a later mutation that adds a block inherits a sensible value)
    kernel     convolution kernel size, {3, 5, 7}
    dense      hidden fully connected units, {0 = none, 32, 64, 128, 256}
    dropout    real, 0.0..0.6 (quantised to 0.05)
    lr_exp     real, learning rate = 10**lr_exp, -4.0..-1.5 (quantised to 0.1)

Decoded network (Neocognitron view in brackets)
    input 1 x 64 x 64
    n_blocks x [ Conv k x k -> BatchNorm -> ReLU   (S-cells: feature extraction)
                 MaxPool 2 x 2                     (C-cells: tolerance to shifts) ]
    global average pool -> Dropout -> [Dense -> ReLU -> Dropout] -> 6 classes
"""

from dataclasses import asdict, dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

FILTER_CHOICES = (8, 16, 32, 64, 128)
KERNEL_CHOICES = (3, 5, 7)
DENSE_CHOICES = (0, 32, 64, 128, 256)
MAX_BLOCKS = 5
DROPOUT_RANGE = (0.0, 0.6)
LR_EXP_RANGE = (-4.0, -1.5)


@dataclass(frozen=True)
class Genome:
    n_blocks: int
    filters: tuple
    kernel: int
    dense: int
    dropout: float
    lr_exp: float

    @property
    def lr(self):
        return 10 ** self.lr_exp

    @property
    def used(self):
        return tuple(self.filters[:self.n_blocks])

    def canonical(self):
        clip = lambda v, r: min(max(v, r[0]), r[1])
        return replace(self, n_blocks=int(clip(self.n_blocks, (1, MAX_BLOCKS))), filters=tuple(self.filters),
                       dropout=round(round(clip(self.dropout, DROPOUT_RANGE) * 20) / 20, 2),
                       lr_exp=round(clip(self.lr_exp, LR_EXP_RANGE), 1))

    def key(self):
        """Identity of the *network* (unused filter genes ignored): used to avoid retraining twins."""
        return (self.used, self.kernel, self.dense, self.dropout, self.lr_exp)

    def describe(self):
        dense = self.dense if self.dense else "none"
        return (f"conv {'-'.join(map(str, self.used))} | kernel {self.kernel}x{self.kernel} | dense {dense} | "
                f"dropout {self.dropout:.2f} | lr {self.lr:.1e}")

    def to_dict(self):
        d = asdict(self)
        d["filters"] = list(self.filters)
        return d

    @staticmethod
    def from_dict(d):
        return Genome(d["n_blocks"], tuple(d["filters"]), d["kernel"], d["dense"], d["dropout"], d["lr_exp"])


# A typical "first guess" a person would write by hand: the baseline the GA must beat.
HAND_DESIGNED = Genome(3, (32, 64, 128, 128, 128), 3, 128, 0.5, -3.0)


class EvolvedCNN(nn.Module):
    def __init__(self, g: Genome, n_classes=6, in_ch=1):
        super().__init__()
        blocks, c = [], in_ch
        for f in g.used:
            blocks.append(nn.Sequential(
                nn.Conv2d(c, f, g.kernel, padding=g.kernel // 2, bias=False),
                nn.BatchNorm2d(f), nn.ReLU(inplace=True), nn.MaxPool2d(2)))
            c = f
        self.features = nn.Sequential(*blocks)
        head = [nn.Flatten(), nn.Dropout(g.dropout)]
        if g.dense:
            head += [nn.Linear(c, g.dense), nn.ReLU(inplace=True), nn.Dropout(g.dropout)]
            c = g.dense
        head.append(nn.Linear(c, n_classes))
        self.head = nn.Sequential(*head)

    def forward(self, x):
        return self.head(F.adaptive_avg_pool2d(self.features(x), 1))


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def grad_cam(model, x, cls=None):
    """Class probabilities and a Grad-CAM heatmap (H, W) in [0, 1]: where the network looked.
    Taken on the last conv block's ReLU output *before* its max-pool (twice the resolution:
    a 5-block net leaves only 2x2 after the final pool, which is too coarse to show anything)."""
    model.eval()
    last = model.features[-1]
    a = last[:-1](model.features[:-1](x))                 # conv -> BN -> ReLU of the last block
    logits = model.head(F.adaptive_avg_pool2d(last[-1](a), 1))
    probs = logits.softmax(1)[0]
    cls = int(probs.argmax()) if cls is None else cls
    g, = torch.autograd.grad(logits[0, cls], a)
    cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam / cam.max() if cam.max() > 0 else cam
    return probs.detach(), cam.detach()
