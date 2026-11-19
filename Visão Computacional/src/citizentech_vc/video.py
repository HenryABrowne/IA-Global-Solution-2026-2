"""Monitoramento contínuo: processamento de vídeo/sequência e geração do vídeo de simulação."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .infer import CrackDetector, draw_analysis


@dataclass
class FrameRecord:
    frame: int
    t_video: float
    prob_max: float
    frac_tiles: float
    score_ema: float
    alerta: bool
    latency_ms: float
    fps: float


def build_mosaic(paths_pos, paths_neg, rows: int = 6, cols: int = 16, tile: int = 227, crack_frac: float = 0.22, seed: int = 0):
    """Monta uma 'fachada' com recortes de TESTE (não vistos no treino).

    As fissuras aparecem em trechos contíguos (como numa parede real), não espalhadas.
    Retorna a imagem RGB e a máscara verdade (rows, cols) com 1 = fissura.
    """
    rng = np.random.default_rng(seed)
    truth = np.zeros((rows, cols), np.uint8)
    n_target = int(rows * cols * crack_frac)
    while truth.sum() < n_target:  # passeio aleatório -> trechos fissurados contínuos
        r, c = rng.integers(0, rows), rng.integers(0, cols)
        for _ in range(rng.integers(3, 8)):
            truth[r, c] = 1
            r = int(np.clip(r + rng.integers(-1, 2), 0, rows - 1))
            c = int(np.clip(c + rng.integers(-1, 2), 0, cols - 1))
    pos = list(rng.permutation(paths_pos))
    neg = list(rng.permutation(paths_neg))
    mosaic = np.zeros((rows * tile, cols * tile, 3), np.uint8)
    for r in range(rows):
        for c in range(cols):
            p = pos.pop() if truth[r, c] else neg.pop()
            im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            mosaic[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile] = cv2.resize(im, (tile, tile))
    return mosaic, truth


def pan_frames(mosaic: np.ndarray, out_w: int = 1024, out_h: int = 576, n_frames: int = 360, field_effects: bool = True, seed: int = 1):
    """Câmera percorrendo a fachada (zigue-zague), com sombra passando, variação de luz e tremor."""
    rng = np.random.default_rng(seed)
    H, W = mosaic.shape[:2]
    lanes = max(1, int(np.ceil((H - out_h) / (out_h * 0.8))) + 1)
    ys = np.linspace(0, H - out_h, lanes).astype(int)
    per_lane = n_frames // lanes
    for li, y in enumerate(ys):
        xs = np.linspace(0, W - out_w, per_lane)
        if li % 2:
            xs = xs[::-1]
        for k, x in enumerate(xs):
            jx, jy = rng.normal(0, 2, 2) if field_effects else (0, 0)
            x0 = int(np.clip(x + jx, 0, W - out_w))
            y0 = int(np.clip(y + jy, 0, H - out_h))
            f = mosaic[y0 : y0 + out_h, x0 : x0 + out_w].astype(np.float32)
            if field_effects:
                t = (li * per_lane + k) / n_frames
                f *= 0.85 + 0.25 * np.sin(2 * np.pi * t * 1.5)  # nuvens / variação de luz
                sx = int((t * 3 % 1) * (out_w + 400)) - 200  # sombra de poste passando
                band = np.ones((out_h, out_w), np.float32)
                band[:, max(0, sx) : max(0, min(out_w, sx + 90))] = 0.5
                f *= cv2.GaussianBlur(band, (0, 0), 6)[..., None]
            yield np.clip(f, 0, 255).astype(np.uint8)


def draw_hud(frame: np.ndarray, rec: FrameRecord, threshold: float, device_name: str) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 34), (20, 24, 32), -1)
    status = "ALERTA: INSPECIONAR" if rec.alerta else "monitorando"
    col = (255, 70, 70) if rec.alerta else (120, 220, 140)
    cv2.putText(out, "CitizenTech | Monitoramento de fissuras", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(out, status, (w - 250, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    info = (
        f"frame {rec.frame}  conf.max {rec.prob_max:.2f}  area sinalizada {rec.frac_tiles:.0%}  "
        f"latencia {rec.latency_ms:.0f} ms  {rec.fps:.1f} FPS  limiar {threshold:.2f}  [{device_name}]"
    )
    cv2.rectangle(out, (0, h - 28), (w, h), (20, 24, 32), -1)
    cv2.putText(out, info, (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return out


def monitor(
    detector: CrackDetector,
    frames,
    threshold: float,
    tile: int | None = None,
    every: int = 1,
    ema_alpha: float = 0.4,
    on_frames: int = 3,
    video_fps: float = 24.0,
):
    """Processa uma sequência de quadros. Gera (quadro anotado, FrameRecord).

    Alerta com histerese: liga após `on_frames` quadros com score >= limiar e
    desliga quando o score médio móvel cai abaixo de 0,8 x limiar. Isso evita
    alertas piscando por um único quadro ruim.
    """
    import torch

    dev = torch.cuda.get_device_name(0) if detector.device == "cuda" else "CPU"
    ema, streak, alert = 0.0, 0, False
    last = None
    t_prev = time.perf_counter()
    fps_ema = 0.0
    for i, frame in enumerate(frames):
        if i % every == 0 or last is None:
            res = detector.analyze(frame, tile=tile)
            last = res
            ema = ema_alpha * res.prob_max + (1 - ema_alpha) * ema
            streak = streak + 1 if res.prob_max >= threshold else 0
            if not alert and streak >= on_frames:
                alert = True
            elif alert and ema < 0.8 * threshold:
                alert = False
        else:
            res = last
        annotated = draw_analysis(frame, res, threshold)
        now = time.perf_counter()
        inst = 1.0 / max(now - t_prev, 1e-6)
        t_prev = now
        fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst
        rec = FrameRecord(i, i / video_fps, res.prob_max, res.frac_tiles, ema, alert, res.latency_ms, fps_ema)
        yield draw_hud(annotated, rec, threshold, dev), rec


def read_video(path: str | Path, max_side: int = 1024, max_frames: int | None = None):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok or (max_frames and len(frames) >= max_frames):
            break
        h, w = f.shape[:2]
        s = min(1.0, max_side / max(h, w))
        if s < 1:
            f = cv2.resize(f, (int(w * s) // 2 * 2, int(h * s) // 2 * 2), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def write_video(frames, path: str | Path, fps: float = 24.0):
    """MP4 H.264 (abre no navegador/Streamlit)."""
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=8, pixelformat="yuv420p") as wr:
        for f in frames:
            wr.append_data(f)
    return path
