"""
Web demo for the GA-designed steel-defect CNN.

  Tab 1  Live defect prediction: upload / pick a steel-surface image -> class probabilities
         and a Grad-CAM heatmap of where the evolved CNN looked; or classify a random test batch
  Tab 2  Evolution results: best and average accuracy per generation, the winning
         architecture, GA vs random search vs hand-designed CNN (from `python evolve.py`)
  Tab 3  Evolve live: run a small GA in front of the audience and watch the curve grow

    python app.py        (opens http://127.0.0.1:7860 in the browser)
"""

import json
import os

import gradio as gr
import matplotlib
import numpy as np
import torch
from PIL import Image

import data as D
import plots
from cnn import EvolvedCNN, Genome, grad_cam
from ga import Evaluator, GAConfig, evolve_iter
from trainer import train_model

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "best_cnn.pt")
RESULTS = os.path.join(HERE, "results")
EXAMPLES = os.path.join(HERE, "examples")
JET = matplotlib.colormaps["jet"]
_cache = {}


def cell(describe):
    """Genome description safe inside a markdown table cell (its ' | ' separators would split the cell)."""
    return describe.replace(" | ", " · ")


def load_model():
    if "model" not in _cache:
        if not os.path.exists(MODEL_PATH):
            raise gr.Error("No trained model yet. Run: python evolve.py")
        ck = torch.load(MODEL_PATH, map_location=D.device(), weights_only=False)
        model = EvolvedCNN(Genome.from_dict(ck["genome"])).to(D.device())
        model.load_state_dict(ck["state_dict"])
        model.eval()
        _cache["model"] = (model, ck)
    return _cache["model"]


def get_data():
    if "data" not in _cache:
        _cache["data"] = D.load_all()
    return _cache["data"]


def ensure_examples(per_class=2):
    """Copy a few held-out TEST images (never seen in training or GA search) into examples/."""
    os.makedirs(EXAMPLES, exist_ok=True)
    out = []
    files = D.list_split("test")
    for c in D.CLASSES:
        for f in [f for f in files if D.label_of(f) == D.CLASSES.index(c)][:per_class]:
            dst = os.path.join(EXAMPLES, os.path.basename(f).replace(".jpg", ".png"))
            if not os.path.exists(dst):
                Image.open(f).convert("L").save(dst)
        out += sorted(os.path.join(EXAMPLES, p) for p in os.listdir(EXAMPLES) if p.startswith(c + "_"))[:per_class]
    if not out:
        out = sorted(os.path.join(EXAMPLES, p) for p in os.listdir(EXAMPLES) if p.endswith(".png"))
    return out


# ------------------------------------------------------------------ tab 1: prediction
def predict(img):
    if img is None:
        raise gr.Error("Upload a steel-surface image or pick an example first.")
    model, ck = load_model()
    x = D.prepare(img, ck["mean"], ck["std"], ck["img"])
    probs, cam = grad_cam(model, x)
    probs = probs.cpu().numpy()
    base = D.load_image(img, 320)
    cam = np.asarray(Image.fromarray((cam.cpu().numpy() * 255).astype(np.uint8)).resize((320, 320), Image.BILINEAR)) / 255
    heat = JET(cam)[..., :3]
    over = (0.55 * np.repeat(base[..., None], 3, 2) + 0.45 * heat) * 255
    k = int(probs.argmax())
    second = int(np.argsort(probs)[-2])
    md = (f"### {D.PRETTY[k]}  ({probs[k] * 100:.1f}% confident)\n"
          f"Runner-up: {D.PRETTY[second]} ({probs[second] * 100:.1f}%).  \n"
          f"Evolved network: `{Genome.from_dict(ck['genome']).describe()}`, "
          f"test accuracy {ck['test_acc'] * 100:.1f}%.  \n"
          f"Heatmap: Grad-CAM of the last convolution block. Red = regions that drove the decision.")
    return {D.PRETTY[i]: float(p) for i, p in enumerate(probs)}, over.astype(np.uint8), md


def random_batch(seed):
    model, ck = load_model()
    files = D.list_split("test")
    rng = np.random.default_rng(int(seed))
    pick = rng.choice(len(files), 12, replace=False)
    x = torch.cat([D.prepare(files[i], ck["mean"], ck["std"], ck["img"]) for i in pick])
    with torch.no_grad():
        p = model(x).softmax(1).cpu().numpy()
    items, right = [], 0
    for i, pr in zip(pick, p):
        t, k = D.label_of(files[i]), int(pr.argmax())
        right += t == k
        mark = "✓" if t == k else f"✗ (true: {D.PRETTY[t]})"
        items.append((np.asarray(Image.open(files[i]).convert("RGB")), f"{D.PRETTY[k]} {pr[k] * 100:.0f}% {mark}"))
    return items, f"**{right} / 12 correct** on randomly drawn held-out test images (seed {int(seed)})."


# ------------------------------------------------------------------ tab 2: saved results
def results_markdown():
    path = os.path.join(RESULTS, "metrics.json")
    if not os.path.exists(path):
        return "No results yet. Run `python evolve.py` first."
    m = json.load(open(path))
    md = (f"**Search cost:** {m['networks_trained_by_ga']} different CNNs trained by the GA in "
          f"{m['ga_minutes']} min on `{m['device']}` ({m['config']['epochs']} epochs each).\n\n"
          "| Model | Test accuracy (mean ± sd over seeds) | Parameters | Architecture |\n|---|---|---|---|\n")
    for s in m["final"]:
        md += (f"| {s['model']} | **{s['test_acc_mean'] * 100:.2f}%** ± {s['test_acc_std'] * 100:.2f} | "
               f"{s['params']:,} | `{cell(s['describe'])}` |\n")
    hist = os.path.join(RESULTS, "ga_history.json")
    if os.path.exists(hist):
        last = json.load(open(hist))["generations"][-1]["population"][:8]
        md += "\n**Final population (top 8):**\n\n| # | Val accuracy | Parameters | Genome |\n|---|---|---|---|\n"
        for i, ind in enumerate(last, 1):
            md += f"| {i} | {ind['acc'] * 100:.2f}% | {ind['params']:,} | `{cell(ind['describe'])}` |\n"
    return md


def result_image(name):
    p = os.path.join(RESULTS, name)
    return p if os.path.exists(p) else None


# ------------------------------------------------------------------ tab 3: live GA
def live_evolve(pop, gens, epochs, seed):
    Dt = get_data()
    pop, gens, epochs, seed = int(pop), int(gens), int(epochs), int(seed)
    cfg = GAConfig(pop_size=pop, generations=gens, seed=seed, elite=min(2, pop - 1))
    ev = Evaluator(lambda g: train_model(g, Dt, epochs, seed=seed)[1])
    head = ("| Gen | Best | Average | Networks trained | Best genome |\n|---|---|---|---|---|\n")
    rows, records = [], []
    yield None, None, f"Training generation 0: {pop} random CNNs, {epochs} epochs each on `{Dt.device}` ..."
    for gen, p in evolve_iter(ev, cfg):
        mean = float(np.mean([i.acc for i in p]))
        records.append({"gen": gen, "accs": [i.acc for i in p], "best_acc": p[0].acc, "mean_acc": mean})
        rows.append(f"| {gen} | {p[0].acc * 100:.1f}% | {mean * 100:.1f}% | {len(ev.trace)} | `{cell(p[0].genome.describe())}` |")
        status = "done." if gen == gens else f"training generation {gen + 1} ..."
        yield (plots.ga_progress(records, title="Live evolution: validation accuracy per generation"),
               plots.architecture(p[0].genome, Dt.size, params=p[0].params, title=f"Best after generation {gen}"),
               head + "\n".join(rows) + f"\n\n*{status}*")


def build():
    examples = ensure_examples()
    with gr.Blocks(title="GA-Designed CNN for Steel Defect Detection") as demo:
        gr.Markdown(
            "# GA-designed CNN for industrial steel-surface defect detection\n"
            "Choosing a CNN by hand is guesswork. Here a **genetic algorithm** evolves the architecture "
            "(number of conv layers, filters per layer, kernel size, dense units) and the training "
            "hyper-parameters (learning rate, dropout). **Fitness = validation accuracy** after a brief "
            "training run. Data: NEU hot-rolled steel strip defects, 6 classes.")
        with gr.Tab("Live defect prediction"):
            with gr.Row():
                with gr.Column(scale=1):
                    img = gr.Image(label="Steel-surface image (grayscale or RGB)", type="numpy", height=300)
                    btn = gr.Button("Classify defect", variant="primary")
                    if examples:
                        gr.Examples(examples, inputs=img, label="Held-out test images (2 per class)",
                                    examples_per_page=12)
                with gr.Column(scale=1):
                    out_label = gr.Label(label="Defect class probabilities", num_top_classes=6)
                    out_md = gr.Markdown()
                with gr.Column(scale=1):
                    out_cam = gr.Image(label="Grad-CAM: where the evolved CNN looked", height=320)
            btn.click(predict, img, [out_label, out_cam, out_md])
            img.upload(predict, img, [out_label, out_cam, out_md])
            gr.Markdown("### Random batch from the test set")
            with gr.Row():
                seed_b = gr.Number(value=0, precision=0, label="seed", scale=1)
                btn_b = gr.Button("Classify 12 random test images", scale=3)
            out_batch_md = gr.Markdown()
            out_gallery = gr.Gallery(label="Predictions (✓ correct, ✗ wrong)", columns=6, height=420)
            btn_b.click(random_batch, seed_b, [out_gallery, out_batch_md])

        with gr.Tab("Evolution results"):
            with gr.Row():
                gr.Image(result_image("ga_progress.png"), label="Best and average accuracy per generation")
                gr.Image(result_image("search_comparison.png"), label="GA vs random search, same budget")
            gr.Image(result_image("architecture.png"), label="Winning architecture")
            gr.Markdown(results_markdown())
            with gr.Row():
                gr.Image(result_image("confusion_matrix.png"), label="Test confusion matrix (GA winner)")
                gr.Image(result_image("test_predictions.png"), label="Test predictions")

        with gr.Tab("Evolve live"):
            gr.Markdown("Run a small GA right now. Each candidate CNN is decoded from its chromosome, "
                        "trained briefly, and scored on the validation set. Generation 0 is random.")
            with gr.Row():
                pop = gr.Slider(4, 20, value=8, step=1, label="population")
                gens = gr.Slider(1, 15, value=5, step=1, label="generations")
                ep = gr.Slider(1, 8, value=3, step=1, label="epochs per candidate")
                sd = gr.Number(value=1, precision=0, label="seed")
            go = gr.Button("Evolve", variant="primary")
            with gr.Row():
                live_plot = gr.Plot(label="Accuracy per generation")
                live_arch = gr.Plot(label="Best architecture so far")
            live_md = gr.Markdown()
            go.click(live_evolve, [pop, gens, ep, sd], [live_plot, live_arch, live_md])
    return demo


if __name__ == "__main__":
    build().launch(inbrowser=True)
