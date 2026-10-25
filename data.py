"""
NEU steel-surface defect images -> normalised tensors on the GPU (or CPU).

The whole dataset is tiny (~1,800 images), so it is loaded once into device memory
and every GA candidate trains from there with no data-loader overhead.
"""

import glob
import os
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "neu")
CLASSES = ("crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches")
PRETTY = ("Crazing", "Inclusion", "Patches", "Pitted surface", "Rolled-in scale", "Scratches")
IMG = 64          # images are 200x200; 64x64 keeps each GA candidate's training to a few seconds


def label_of(path):
    return CLASSES.index(os.path.basename(path).rsplit("_", 1)[0])      # "pitted_surface_12.jpg"


def list_split(split):
    return sorted(glob.glob(os.path.join(DATA, split, "*.jpg")))


def load_image(img, size=IMG):
    """File path or numpy array (gray or RGB) -> float32 array (size, size) in [0, 1]."""
    im = Image.open(img) if isinstance(img, str) else Image.fromarray(np.asarray(img).astype(np.uint8))
    return np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), np.float32) / 255.0


def load_split(split, size=IMG):
    files = list_split(split)
    if not files:
        raise SystemExit(f"No images in {os.path.join(DATA, split)}. Run: python download_data.py")
    X = np.stack([load_image(f, size) for f in files])[:, None]          # (N, 1, H, W)
    y = np.array([label_of(f) for f in files])
    return X, y, files


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_all(size=IMG, dev=None):
    """Train / val / test tensors, normalised with the training-set mean and std."""
    dev = dev or device()
    Xtr, ytr, ftr = load_split("train", size)
    Xva, yva, fva = load_split("val", size)
    Xte, yte, fte = load_split("test", size)
    mean, std = float(Xtr.mean()), float(Xtr.std())
    t = lambda X: torch.as_tensor((X - mean) / std, device=dev)
    l = lambda y: torch.as_tensor(y, device=dev)
    return SimpleNamespace(Xtr=t(Xtr), ytr=l(ytr), Xva=t(Xva), yva=l(yva), Xte=t(Xte), yte=l(yte),
                           files_tr=ftr, files_va=fva, files_te=fte,
                           mean=mean, std=std, size=size, device=dev)


def augment(x):
    """Random flips / transposes (all 8 symmetries of the square): defects have no fixed orientation."""
    n = len(x)
    flip = lambda m: m.view(n, 1, 1, 1)
    x = torch.where(flip(torch.rand(n, device=x.device) < 0.5), x.flip(3), x)
    x = torch.where(flip(torch.rand(n, device=x.device) < 0.5), x.flip(2), x)
    x = torch.where(flip(torch.rand(n, device=x.device) < 0.5), x.transpose(2, 3), x)
    return x


def prepare(img, mean, std, size=IMG, dev=None):
    """Single image (path or array) -> normalised tensor (1, 1, size, size)."""
    return torch.as_tensor((load_image(img, size) - mean) / std, device=dev or device())[None, None]
