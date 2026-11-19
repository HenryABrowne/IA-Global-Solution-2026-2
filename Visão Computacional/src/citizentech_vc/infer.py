"""Inferência para o CitizenTech: recorte único, imagem grande (janelas), Grad-CAM e qualidade."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from .model import build_model, gradcam_layer, to_tensor_batch

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


@dataclass
class TileResult:
    x: int
    y: int
    w: int
    h: int
    prob: float


@dataclass
class ImageAnalysis:
    prob_max: float
    frac_tiles: float
    n_tiles: int
    tiles: list[TileResult] = field(default_factory=list)
    latency_ms: float = 0.0
    tile_size: int = 0

    def decision(self, threshold: float) -> str:
        return "INSPEÇÃO RECOMENDADA" if self.prob_max >= threshold else "Sem indício de fissura"


class CrackDetector:
    """Carrega o modelo exportado pelo notebook (models/<arch>.pt + .json)."""

    def __init__(self, arch: str | None = None, models_dir: str | Path = MODELS_DIR, device: str | None = None, half: bool = False):
        models_dir = Path(models_dir)
        if arch is None:
            arch = json.loads((models_dir / "selected.json").read_text(encoding="utf-8"))["arch"]
        self.cfg = json.loads((models_dir / f"{arch}.json").read_text(encoding="utf-8"))
        self.name = arch
        self.arch = self.cfg.get("arch", arch)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.half = half and self.device == "cuda"
        self.size = int(self.cfg.get("img_size", 224))
        self.temperature = float(self.cfg.get("temperature", 1.0))
        self.thresholds = self.cfg.get("thresholds", {"triagem": 0.5})
        m = build_model(self.arch, pretrained=False)
        m.load_state_dict(torch.load(models_dir / f"{arch}.pt", map_location="cpu", weights_only=True))
        self.model = m.to(self.device).eval()
        if self.half:
            self.model.half()

    # ------------------------------------------------------------ básico
    def _prep(self, crops_rgb: list[np.ndarray]) -> torch.Tensor:
        arr = np.stack([cv2.resize(c, (self.size, self.size), interpolation=cv2.INTER_AREA) for c in crops_rgb])
        t = to_tensor_batch(arr, self.device)
        return t.half() if self.half else t

    @torch.no_grad()
    def predict_crops(self, crops_rgb: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(crops_rgb), batch_size):
            logits = self.model(self._prep(crops_rgb[i : i + batch_size])).float()
            out.append(torch.softmax(logits / self.temperature, 1)[:, 1].cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    # ------------------------------------------------------- imagem grande
    @staticmethod
    def auto_tile(h: int, w: int) -> int:
        """Recortes da base cobrem ~1/13 da menor dimensão da foto original."""
        if min(h, w) <= 320:
            return min(h, w)
        return int(np.clip(min(h, w) / 6, 160, 480))

    def analyze(self, img_rgb: np.ndarray, tile: int | None = None, overlap: float = 0.25) -> ImageAnalysis:
        t0 = time.perf_counter()
        h, w = img_rgb.shape[:2]
        tile = tile or self.auto_tile(h, w)
        tile = min(tile, h, w)
        stride = max(1, int(tile * (1 - overlap)))
        ys = list(range(0, max(h - tile, 0) + 1, stride))
        xs = list(range(0, max(w - tile, 0) + 1, stride))
        if ys[-1] + tile < h:
            ys.append(h - tile)
        if xs[-1] + tile < w:
            xs.append(w - tile)
        coords = [(x, y) for y in ys for x in xs]
        crops = [img_rgb[y : y + tile, x : x + tile] for x, y in coords]
        probs = self.predict_crops(crops)
        if self.device == "cuda":
            torch.cuda.synchronize()
        tiles = [TileResult(x, y, tile, tile, float(p)) for (x, y), p in zip(coords, probs)]
        thr = self.thresholds.get("triagem", 0.5)
        return ImageAnalysis(
            prob_max=float(probs.max()),
            frac_tiles=float((probs >= thr).mean()),
            n_tiles=len(tiles),
            tiles=tiles,
            latency_ms=(time.perf_counter() - t0) * 1000,
            tile_size=tile,
        )

    # ----------------------------------------------------------- Grad-CAM
    def gradcam(self, crop_rgb: np.ndarray) -> np.ndarray:
        """Mapa de calor (H,W) em [0,1] das regiões que levaram à classe fissura."""
        model = self.model.float()
        layer = gradcam_layer(model, self.arch)
        store = {}
        h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
        try:
            x = to_tensor_batch(cv2.resize(crop_rgb, (self.size, self.size))[None], self.device)
            x.requires_grad_(True)
            with torch.enable_grad():
                logits = model(x)
                act = store["act"]
                grad = torch.autograd.grad(logits[0, 1], act)[0]
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * act).sum(1))[0].detach().cpu().numpy()
        finally:
            h1.remove()
            if self.half:
                self.model.half()
        cam = cv2.resize(cam, (crop_rgb.shape[1], crop_rgb.shape[0]))
        return cam / (cam.max() + 1e-8)


# ------------------------------------------------------------- qualidade
def image_quality(img_rgb: np.ndarray) -> dict:
    """Checagens antes de confiar no modelo (nitidez pelo Laplaciano, exposição)."""
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    s = 1024 / max(g.shape)
    if s < 1:  # só reduz; ampliar borraria a imagem e falsearia a medida
        g = cv2.resize(g, (int(g.shape[1] * s), int(g.shape[0] * s)), interpolation=cv2.INTER_AREA)
    # Nitidez RELATIVA: var(Laplaciano) da imagem / var(Laplaciano) dela levemente borrada.
    # Concreto liso tem pouca textura, então o valor absoluto engana; a razão não.
    # Calibrado em 400 recortes de teste: nítidas p5 = 4,9; desfoque gaussiano p95 = 1,9.
    lap = cv2.Laplacian(g, cv2.CV_64F).var()
    sharp = float(lap / (cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1.5), cv2.CV_64F).var() + 1e-6))
    bright = float(g.mean())
    contrast = float(g.std())
    alerts = []
    if sharp < 3.0:
        alerts.append("Imagem possivelmente borrada (nitidez relativa baixa)")
    if bright < 50:
        alerts.append("Imagem escura: risco de fissuras não detectadas")
    if bright > 215:
        alerts.append("Imagem superexposta")
    if contrast < 15:
        alerts.append("Baixo contraste")
    return {"nitidez": sharp, "brilho": bright, "contraste": contrast, "alertas": alerts}


# ------------------------------------------------------------ desenho
def draw_analysis(img_rgb: np.ndarray, res: ImageAnalysis, threshold: float, heat_alpha: float = 0.35) -> np.ndarray:
    """Sobrepõe mapa de probabilidade + caixas nas janelas acima do limiar."""
    h, w = img_rgb.shape[:2]
    heat = heatmap_from_tiles(res, img_rgb.shape)
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    mask = (heat >= threshold)[..., None]
    out = np.where(mask, (img_rgb * (1 - heat_alpha) + color * heat_alpha).astype(np.uint8), img_rgb)
    lw = max(2, int(min(h, w) / 300))
    for (x, y, bw, bh), p in damage_regions(heat, threshold):
        cv2.rectangle(out, (x, y), (x + bw - 1, y + bh - 1), (255, 40, 40), lw)
        label = f"fissura {p:.0%}"
        fs = 0.45 * lw / 2 + 0.2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, lw // 2))
        cv2.rectangle(out, (x, y), (x + tw + 6, y + th + 8), (255, 40, 40), -1)
        cv2.putText(out, label, (x + 3, y + th + 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, lw // 2), cv2.LINE_AA)
    return out


def damage_regions(heat: np.ndarray, threshold: float):
    """Regiões contíguas acima do limiar -> caixas (x, y, w, h) + prob. máxima."""
    mask = (heat >= threshold).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = []
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        regions.append(((int(x), int(y), int(bw), int(bh)), float(heat[lab == i].max())))
    return sorted(regions, key=lambda r: -r[1])


def heatmap_from_tiles(res: ImageAnalysis, shape) -> np.ndarray:
    h, w = shape[:2]
    heat = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    for t in res.tiles:
        heat[t.y : t.y + t.h, t.x : t.x + t.w] += t.prob
        cnt[t.y : t.y + t.h, t.x : t.x + t.w] += 1
    return heat / np.maximum(cnt, 1)


def overlay_cam(crop_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return (crop_rgb * (1 - alpha) + color * alpha).astype(np.uint8)
