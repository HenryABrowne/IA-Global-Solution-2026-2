"""Preparação dos dados: amostragem, grupos por imagem de origem, split e cache.

A base "Concrete Crack Images for Classification" (Özgenel, 2018) tem 40 mil
recortes 227x227 gerados a partir de 458 fotos de alta resolução. Os arquivos
não informam de qual foto veio cada recorte, então reconstruímos grupos
aproximados para que recortes da mesma foto não fiquem em conjuntos diferentes:

* Positive: os arquivos são numerados na ordem em que foram recortados.
  Recortes vizinhos na numeração são muito parecidos (ver notebook), então
  blocos contíguos de numeração formam um grupo.
* Negative: a numeração foi embaralhada (sem correlação com a vizinhança).
  Agrupamos por aparência (cor e textura em Lab) com K-Means: recortes da
  mesma superfície caem no mesmo cluster.

O split treino/validação/teste é feito por grupo, nunca por imagem.
"""
from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DEFAULT_ROOT = os.environ.get("CRACK_DATA", r"C:/Users/lftra/datasets/concrete_crack")


def file_key(p: Path) -> tuple[int, int]:
    """'10000_1.jpg' -> (1, 10000); '00042.jpg' -> (0, 42).

    A pasta Positive tem duas sequências de recorte: 00001..10621 e 10000_1..19378_1.
    (Cuidado: int('19378_1') em Python vale 193781, por isso o parse explícito.)
    """
    base, _, suf = p.stem.partition("_")
    return (int(suf) if suf else 0, int(base))


def list_images(root: str | Path = DEFAULT_ROOT):
    root = Path(root)
    pos = sorted((root / "Positive").glob("*.jpg"), key=file_key)
    neg = sorted((root / "Negative").glob("*.jpg"), key=file_key)
    if not pos or not neg:
        raise FileNotFoundError(
            f"Base não encontrada em {root}. Esperado {root}/Positive e {root}/Negative. "
            "Defina a variável de ambiente CRACK_DATA."
        )
    return pos, neg


def appearance_features(path: Path, size: int = 64) -> np.ndarray:
    """Média/desvio em Lab + energia de bordas: assinatura barata da superfície."""
    im = cv2.imread(str(path))
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return np.r_[lab.mean(0), lab.std(0), np.abs(lap).mean(), lap.std()]


def neighbour_distance_profile(paths, gaps=(1, 2, 5, 10, 50, 200, 1000)):
    """Distância média de aparência entre arquivos separados por `gap` na numeração."""
    with ThreadPoolExecutor(8) as ex:
        f = np.array(list(ex.map(appearance_features, paths)))
    f = (f - f.mean(0)) / (f.std(0) + 1e-6)
    return {g: float(np.linalg.norm(f[g:] - f[:-g], axis=1).mean()) for g in gaps if g < len(f)}


def build_sample(
    root: str | Path = DEFAULT_ROOT,
    n_per_class: int = 6000,
    pos_block: int = 100,
    neg_clusters: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """Amostra balanceada com coluna `group` (imagem de origem aproximada)."""
    rng = np.random.default_rng(seed)
    pos, neg = list_images(root)

    # Positivos: blocos contíguos inteiros (não quebra grupos ao amostrar)
    keys = [file_key(p) for p in pos]
    blocks = [f"P{s}-{(n - 1) // pos_block:03d}" for s, n in keys]
    ub = np.array(sorted(set(blocks)))
    rng.shuffle(ub)
    sizes = pd.Series(blocks).value_counts()
    chosen, acc = set(), 0
    for b in ub:  # blocos inteiros até atingir n_per_class
        if acc >= n_per_class:
            break
        chosen.add(b)
        acc += sizes[b]
    pos_rows = [(str(p), 1, b) for p, b in zip(pos, blocks) if b in chosen]

    # Negativos: amostra aleatória + clusters de aparência como grupos
    neg_sel = [neg[i] for i in rng.choice(len(neg), size=n_per_class, replace=False)]
    with ThreadPoolExecutor(8) as ex:
        feats = np.array(list(ex.map(appearance_features, neg_sel)))
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    km = KMeans(n_clusters=neg_clusters, n_init=4, random_state=seed).fit(feats)
    neg_rows = [(str(p), 0, f"N{c:04d}") for p, c in zip(neg_sel, km.labels_)]

    df = pd.DataFrame(pos_rows + neg_rows, columns=["path", "label", "group"])
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split_by_group(df: pd.DataFrame, fracs=(0.70, 0.15, 0.15), seed: int = 42) -> pd.DataFrame:
    """Atribui train/val/test por grupo, estratificado por classe."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["split"] = ""
    names = ["train", "val", "test"]
    for label in sorted(df.label.unique()):
        sizes = df[df.label == label].groupby("group").size()
        groups = sizes.index.to_numpy().copy()
        rng.shuffle(groups)
        total = sizes.sum()
        cuts = np.cumsum(fracs)[:-1] * total
        acc, assign = 0, {}
        for g in groups:
            k = int(np.searchsorted(cuts, acc, side="right"))
            assign[g] = names[k]
            acc += sizes[g]
        mask = df.label == label
        df.loc[mask, "split"] = df.loc[mask, "group"].map(assign)
    # Garantia: nenhum grupo em mais de um conjunto
    assert (df.groupby("group").split.nunique() == 1).all()
    return df


def _load_rgb(path: str, size: int) -> np.ndarray:
    im = cv2.imread(path)
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def load_images(paths, size: int = 224, cache_dir: str | Path | None = None) -> np.ndarray:
    """Carrega imagens RGB uint8 (N, size, size, 3), com cache .npy opcional."""
    paths = list(map(str, paths))
    cache_file = None
    if cache_dir:
        key = hashlib.md5(("|".join(paths) + str(size)).encode()).hexdigest()[:12]
        cache_file = Path(cache_dir) / f"cache_{size}_{key}.npy"
        if cache_file.exists():
            return np.load(cache_file)
    with ThreadPoolExecutor(8) as ex:
        arr = np.stack(list(ex.map(lambda p: _load_rgb(p, size), paths)))
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, arr)
    return arr
