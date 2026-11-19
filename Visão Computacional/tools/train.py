"""Treina as 3 arquiteturas fora do notebook (log em tempo real + checkpoint por modelo).

O notebook usa a mesma amostra/split/configuração e carrega estes checkpoints
(models/ckpt/) se existirem; sem eles, treina dentro do próprio notebook.

    python tools/train.py            # treina o que faltar
    python tools/train.py --force    # retreina tudo
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from citizentech_vc import data, model as M  # noqa: E402
from citizentech_vc.config import CONFIGS, IMG, N_PER_CLASS, SEED  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--force", action="store_true")
args = ap.parse_args()

DATA = Path(os.environ.get("CRACK_DATA", data.DEFAULT_ROOT))
torch.backends.cudnn.benchmark = True
df = data.split_by_group(data.build_sample(DATA, n_per_class=N_PER_CLASS, seed=SEED), seed=SEED)
X = {s: data.load_images(df[df.split == s].path, IMG, cache_dir=DATA.parent / "cache_citizentech") for s in ["train", "val"]}
Y = {s: df[df.split == s].label.to_numpy() for s in ["train", "val"]}
print({s: X[s].shape for s in X}, flush=True)
for arch, cfg in CONFIGS.items():
    _, _, best, secs = M.train_or_load(arch, ROOT / "models" / "ckpt", X["train"], Y["train"], X["val"], Y["val"],
                                        force=args.force, device="cuda" if torch.cuda.is_available() else "cpu",
                                        seed=SEED, log=lambda s: print(s, flush=True), **cfg)
    print(f"  -> {arch}: melhor época {best}, {secs / 60:.1f} min", flush=True)
    torch.cuda.empty_cache()
