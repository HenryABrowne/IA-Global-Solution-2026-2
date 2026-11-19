"""Medição de latência e FPS."""
from __future__ import annotations

import platform
import time

import numpy as np
import torch

from .model import build_model


def hardware_info() -> dict:
    import psutil

    info = {
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "so": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["vram_gb"] = round(p.total_memory / 2**30, 1)
    return info


@torch.no_grad()
def bench_model(arch: str, device: str, res: int = 224, batch: int = 1, half: bool = False,
                n_iter: int = 100, warmup: int = 15, model=None, threads: int | None = None) -> dict:
    """Latência só do modelo (tensor já pronto), média e p95 por lote e por imagem."""
    if threads:
        torch.set_num_threads(threads)
    import copy

    m = (copy.deepcopy(model) if model is not None else build_model(arch, pretrained=False)).float().to(device).eval()
    if half:
        m = m.half()
    x = torch.randn(batch, 3, res, res, device=device)
    if half:
        x = x.half()
    valid = bool(torch.isfinite(m(x)).all().item())
    times = []
    for i in range(warmup + n_iter):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        m(x)
        if device == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    t = np.array(times) * 1000
    return {
        "arquitetura": arch,
        "dispositivo": device,
        "precisao": "fp16" if half else "fp32",
        "resolucao": res,
        "lote": batch,
        "lat_lote_ms": float(t.mean()),
        "lat_p95_ms": float(np.percentile(t, 95)),
        "lat_img_ms": float(t.mean() / batch),
        "imgs_por_s": float(1000 * batch / t.mean()),
        "saida_valida": valid,
    }
