"""Perturbações de campo para testar robustez (sombra, iluminação, desfoque, ruído, obstáculo)."""
from __future__ import annotations

import cv2
import numpy as np


def shadow(img: np.ndarray, rng: np.random.Generator, strength: float = 0.55) -> np.ndarray:
    """Sombra de borda dura (poste, marquise) cruzando o recorte."""
    h, w = img.shape[:2]
    p1 = (int(rng.integers(0, w)), 0)
    p2 = (int(rng.integers(0, w)), h)
    side = rng.integers(0, 2)
    pts = np.array([p1, p2, (w if side else 0, h), (w if side else 0, 0)], np.int32)
    mask = np.zeros((h, w), np.float32)
    cv2.fillPoly(mask, [pts], 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), 3)[..., None]
    return np.clip(img * (1 - strength * mask), 0, 255).astype(np.uint8)


def dark(img, rng, factor: float = 0.35):
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def overexposed(img, rng, gain: float = 1.8, offset: float = 40):
    return np.clip(img.astype(np.float32) * gain + offset, 0, 255).astype(np.uint8)


def blur(img, rng, k: int = 9):
    return cv2.GaussianBlur(img, (k, k), 0)


def motion_blur(img, rng, k: int = 15):
    kern = np.zeros((k, k), np.float32)
    kern[k // 2, :] = 1.0 / k
    m = cv2.getRotationMatrix2D((k / 2, k / 2), float(rng.uniform(0, 180)), 1)
    kern = cv2.warpAffine(kern, m, (k, k))
    kern /= kern.sum()
    return cv2.filter2D(img, -1, kern)


def noise(img, rng, sigma: float = 25):
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def jpeg(img, rng, quality: int = 15):
    ok, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]


def occlusion(img, rng, frac: float = 0.35):
    """Obstáculo opaco (folha, poste, lixo) cobrindo parte do recorte."""
    out = img.copy()
    h, w = img.shape[:2]
    oh, ow = int(h * np.sqrt(frac)), int(w * np.sqrt(frac))
    y, x = int(rng.integers(0, h - oh)), int(rng.integers(0, w - ow))
    color = rng.integers(20, 90, 3)
    out[y : y + oh, x : x + ow] = color
    return out


def low_res(img, rng, factor: int = 4):
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


PERTURBATIONS = {
    "original": lambda img, rng: img,
    "sombra": shadow,
    "escura": dark,
    "superexposta": overexposed,
    "desfoque": blur,
    "borrão de movimento": motion_blur,
    "ruído": noise,
    "JPEG forte": jpeg,
    "obstáculo": occlusion,
    "baixa resolução": low_res,
}


def apply_all(x: np.ndarray, name: str, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fn = PERTURBATIONS[name]
    return np.stack([fn(im, rng) for im in x])
