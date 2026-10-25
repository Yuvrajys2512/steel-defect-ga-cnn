"""
Evolve a CNN for NEU steel-surface defect classification with a genetic algorithm,
then compare it with random search (same compute) and a hand-designed CNN.

    python evolve.py                        # full run: GA 16 x 12 generations, 5 epochs per candidate
    python evolve.py --pop 10 --gens 6      # smaller search
    python evolve.py --quick                # 2-minute smoke test

Steps
  1. GA search      every candidate trained briefly; fitness = validation accuracy
  2. random search  same number of networks trained, random genomes (is the GA better than luck?)
  3. final training GA winner, random-search winner and hand-designed CNN, each trained longer
                    with several seeds; test accuracy measured once at the end
Writes models/best_cnn.pt, results/*.json and results/*.png
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import data as D
import plots
from cnn import HAND_DESIGNED, Genome
from ga import Evaluator, GAConfig, evolve_iter, random_search
from trainer import evaluate, train_model

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
MODELS = os.path.join(HERE, "models")
torch.backends.cudnn.benchmark = True


def generation_record(gen, pop):
    return {"gen": gen, "best_acc": pop[0].acc, "mean_acc": float(np.mean([i.acc for i in pop])),
            "accs": [i.acc for i in pop], "best": pop[0].to_dict(),
            "population": [i.to_dict() for i in pop]}


def run_ga(Dt, cfg, epochs, log=print):
    ev = Evaluator(lambda g: train_model(g, Dt, epochs, seed=cfg.seed)[1])
    gens, t0 = [], time.time()
    for gen, pop in evolve_iter(ev, cfg):
        gens.append(generation_record(gen, pop))
        log(f"  gen {gen:2d}  best {pop[0].acc * 100:6.2f}%  mean {gens[-1]['mean_acc'] * 100:6.2f}%  "
            f"trained so far {len(ev.trace):3d}  ({time.time() - t0:5.0f} s)  best: {pop[0].genome.describe()}")
    return gens, ev


def final_train(name, genome, Dt, epochs, seeds, log=print):
    """Train `genome` with several seeds; keep the model of the first seed for the demo."""
    runs, keep = [], None
    for s in range(seeds):
        model, r = train_model(genome, Dt, epochs, seed=s)
        te_acc, te_loss, probs = evaluate(model, Dt.Xte, Dt.yte)
        runs.append({"seed": s, "val_acc": r["acc"], "test_acc": te_acc, "test_loss": te_loss,
                     "best_epoch": r["best_epoch"], "seconds": round(r["seconds"], 1)})
        log(f"  {name:<22} seed {s}  val {r['acc'] * 100:6.2f}%  test {te_acc * 100:6.2f}%  ({r['seconds']:.0f} s)")
        if keep is None:
            keep = (model, r, probs)
    te = np.array([x["test_acc"] for x in runs])
    summary = {"model": name, "genome": genome.to_dict(), "describe": genome.describe(), "params": keep[1]["params"],
               "test_acc_mean": float(te.mean()), "test_acc_std": float(te.std()),
               "val_acc_mean": float(np.mean([x["val_acc"] for x in runs])), "runs": runs}
    return summary, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=5, help="training epochs per GA candidate (brief)")
    ap.add_argument("--final-epochs", type=int, default=40)
    ap.add_argument("--final-seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-random", action="store_true", help="skip the random-search baseline")
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    args = ap.parse_args()
    if args.quick:
        args.pop, args.gens, args.epochs, args.final_epochs, args.final_seeds = 6, 2, 2, 4, 1

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    Dt = D.load_all()
    print(f"NEU: {len(Dt.Xtr)} train, {len(Dt.Xva)} val, {len(Dt.Xte)} test images at {Dt.size}x{Dt.size}; "
          f"device: {Dt.device}" + (f" ({torch.cuda.get_device_name()})" if Dt.device.type == "cuda" else ""))

    # ---- 1. GA
    cfg = GAConfig(pop_size=args.pop, generations=args.gens, seed=args.seed)
    print(f"\n1. Genetic algorithm: population {cfg.pop_size}, {cfg.generations} generations, "
          f"{args.epochs} epochs per candidate")
    t0 = time.time()
    gens, ev = run_ga(Dt, cfg, args.epochs)
    ga_seconds = time.time() - t0
    ga_best = Genome.from_dict(gens[-1]["best"]["genome"])
    budget = len(ev.trace)
    print(f"  {budget} different networks trained in {ga_seconds / 60:.1f} min "
          f"({sum(1 for g in gens for _ in g['population']) - budget} evaluations saved by the cache)")

    # ---- 2. random search with the same budget
    rs_best, rs_trace = None, []
    if not args.skip_random:
        print(f"\n2. Random search: {budget} random networks, {args.epochs} epochs each")
        rev = Evaluator(lambda g: train_model(g, Dt, args.epochs, seed=args.seed)[1])
        t0 = time.time()
        rs = random_search(rev, budget, seed=args.seed)
        rs_best, rs_trace = rs.genome, rev.trace
        print(f"  best {rs.acc * 100:.2f}%  ({(time.time() - t0) / 60:.1f} min)  {rs.genome.describe()}")

    # ---- 3. final training and test
    print(f"\n3. Final training: {args.final_epochs} epochs x {args.final_seeds} seeds, then the test set")
    contenders = [("GA winner", ga_best), ("Hand-designed CNN", HAND_DESIGNED)]
    if rs_best is not None:
        contenders.insert(1, ("Random-search winner", rs_best))
    finals, kept = [], {}
    for name, g in contenders:
        s, k = final_train(name, g, Dt, args.final_epochs, args.final_seeds)
        finals.append(s)
        kept[name] = k

    model, r, probs = kept["GA winner"]
    pred = probs.argmax(1).cpu().numpy()
    true = Dt.yte.cpu().numpy()
    cm = np.zeros((len(D.CLASSES), len(D.CLASSES)), int)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    torch.save({"state_dict": model.state_dict(), "genome": ga_best.to_dict(), "mean": Dt.mean, "std": Dt.std,
                "img": Dt.size, "classes": D.CLASSES, "test_acc": float((pred == true).mean()),
                "history": r["history"]}, os.path.join(MODELS, "best_cnn.pt"))

    print(f"\n{'Model':<22}{'Test acc':>16}{'Val acc':>10}{'Params':>10}  Architecture")
    print("-" * 110)
    for s in finals:
        print(f"{s['model']:<22}{s['test_acc_mean'] * 100:>9.2f} ± {s['test_acc_std'] * 100:.2f}"
              f"{s['val_acc_mean'] * 100:>9.2f}%{s['params']:>10,}  {s['describe']}")

    # ---- save
    with open(os.path.join(RESULTS, "ga_history.json"), "w") as f:
        json.dump({"config": vars(args), "generations": gens,
                   "ga_trace": [i.to_dict() for i in ev.trace],
                   "random_trace": [i.to_dict() for i in rs_trace]}, f, indent=1)
    per_class = {D.CLASSES[i]: float(cm[i, i] / cm[i].sum()) for i in range(len(D.CLASSES))}
    with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
        json.dump({"config": vars(args), "device": str(Dt.device), "ga_minutes": round(ga_seconds / 60, 1),
                   "networks_trained_by_ga": budget, "final": finals, "confusion_matrix": cm.tolist(),
                   "ga_winner_per_class_test_acc": per_class}, f, indent=2)

    plots.ga_progress(gens).savefig(os.path.join(RESULTS, "ga_progress.png"), dpi=140)
    plots.architecture(ga_best, Dt.size, params=r["params"]).savefig(os.path.join(RESULTS, "architecture.png"), dpi=140)
    plots.confusion(cm).savefig(os.path.join(RESULTS, "confusion_matrix.png"), dpi=140)
    if rs_trace:
        plots.search_comparison([i.acc for i in ev.trace], [i.acc for i in rs_trace]).savefig(
            os.path.join(RESULTS, "search_comparison.png"), dpi=140)
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.where(true == c)[0], 2, replace=False) for c in range(len(D.CLASSES))])
    conf = probs.max(1).values.cpu().numpy()
    imgs = [D.load_image(Dt.files_te[i], 200) for i in idx]
    plots.prediction_grid(imgs, true[idx], pred[idx], conf[idx]).savefig(
        os.path.join(RESULTS, "test_predictions.png"), dpi=140)
    print("\nSaved models/best_cnn.pt, results/metrics.json, results/ga_history.json and results/*.png")


if __name__ == "__main__":
    main()
