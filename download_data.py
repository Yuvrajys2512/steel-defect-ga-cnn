"""
Download the NEU steel-surface defect dataset (Song & Yan, Northeastern University)
from a public Hugging Face mirror, so no Kaggle login is needed.

Mirror: KeenForgeAI/NEU-DET-corrected. Every image pixel is identical to the original
release; 3 duplicate images were removed (1,797 images left) and a seeded,
class-stratified 80/10/10 train/val/test split is provided.

Six defect classes of hot-rolled steel strip, 200x200 grayscale, ~300 per class:
crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches.
The class is in the file name (e.g. pitted_surface_12.jpg).

    python download_data.py          # ~1,800 images, ~30 MB -> data/neu/{train,val,test}/
"""

import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = "KeenForgeAI/NEU-DET-corrected"
API = f"https://huggingface.co/api/datasets/{REPO}"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "neu")


def main():
    with urllib.request.urlopen(API) as r:
        files = [s["rfilename"] for s in json.load(r)["siblings"]]
    todo = [f for f in files if f.startswith("images/") and f.lower().endswith(".jpg")]
    print(f"downloading {len(todo)} images to {OUT}")

    def fetch(f):
        dst = os.path.join(OUT, *f.split("/")[1:])          # images/train/x.jpg -> data/neu/train/x.jpg
        if os.path.exists(dst):
            return None
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        for attempt in range(4):
            try:
                urllib.request.urlretrieve(RAW + urllib.parse.quote(f), dst + ".part")
                os.replace(dst + ".part", dst)               # never leave half-written files behind
                return None
            except Exception as e:                           # flaky network: back off and retry
                err = e
                time.sleep(2 * (attempt + 1))
        return f"{f}: {err}"

    failed = []
    with ThreadPoolExecutor(8) as ex:
        for i, res in enumerate(ex.map(fetch, todo), 1):
            if res:
                failed.append(res)
            if i % 200 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    if failed:
        print(f"{len(failed)} files failed (run the script again to retry):")
        for f in failed[:10]:
            print("  ", f)
    else:
        for split in ("train", "val", "test"):
            d = os.path.join(OUT, split)
            print(f"  {split:5s} {len(os.listdir(d)) if os.path.isdir(d) else 0} images")
        print("done")


if __name__ == "__main__":
    main()
