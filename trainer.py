"""
Train one CNN (decoded from a genome) on the NEU data that already sits on the device.

Used twice:
  * inside the GA, briefly (a few epochs): the resulting validation accuracy is the fitness
  * for the final winner, longer, and then scored once on the untouched test set
"""

import copy
import time

import torch
import torch.nn.functional as F

from cnn import EvolvedCNN, n_params
from data import augment

BATCH = 32


@torch.no_grad()
def evaluate(model, X, y, bs=256):
    model.eval()
    logits = torch.cat([model(X[i:i + bs]) for i in range(0, len(X), bs)])
    loss = F.cross_entropy(logits, y).item()
    probs = logits.softmax(1)
    acc = (probs.argmax(1) == y).float().mean().item()
    return acc, loss, probs


def train_model(genome, D, epochs, seed=0, log=None):
    """Adam + cosine learning-rate decay. Returns the weights of the epoch with the best
    validation accuracy (ties broken by validation loss) and a result dict."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = EvolvedCNN(genome).to(D.device)
    opt = torch.optim.Adam(model.parameters(), lr=genome.lr)
    n = len(D.Xtr)
    steps = epochs * ((n + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    best_key, best_state, best_ep, history = None, None, 0, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).to(D.device)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            loss = F.cross_entropy(model(augment(D.Xtr[idx])), D.ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        va_acc, va_loss, _ = evaluate(model, D.Xva, D.yva)
        if va_loss != va_loss:                     # NaN: learning rate too high, the net diverged
            va_loss = 99.0
        history.append({"train_loss": tot / n, "val_acc": va_acc, "val_loss": va_loss})
        key = (va_acc, -va_loss)
        if best_key is None or key > best_key:
            best_key, best_ep = key, ep
            best_state = copy.deepcopy(model.state_dict())
        if log:
            log(f"    epoch {ep:3d}  train loss {tot / n:.3f}  val acc {va_acc * 100:6.2f}%  val loss {va_loss:.3f}")
    model.load_state_dict(best_state)
    return model, {"acc": best_key[0], "loss": -best_key[1], "params": n_params(model),
                   "seconds": time.time() - t0, "best_epoch": best_ep, "history": history}
