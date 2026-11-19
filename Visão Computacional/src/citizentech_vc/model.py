"""Arquiteturas (transferência de aprendizado), treino e avaliação."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from sklearn import metrics as skm

from . import CLASSES, IMAGENET_MEAN, IMAGENET_STD

ARCHS = {
    # nome: (construtor, pesos ImageNet, camada alvo do Grad-CAM)
    "mobilenet_v3_large": (tvm.mobilenet_v3_large, tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2),
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1),
    "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.IMAGENET1K_V1),
}


def build_model(arch: str, pretrained: bool = True, n_classes: int = 2) -> nn.Module:
    ctor, weights = ARCHS[arch]
    m = ctor(weights=weights if pretrained else None)
    if arch.startswith("mobilenet"):
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
    elif arch.startswith("resnet"):
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif arch.startswith("efficientnet"):
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    return m


def gradcam_layer(model: nn.Module, arch: str) -> nn.Module:
    if arch.startswith("resnet"):
        return model.layer4
    return model.features[-1]


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


# ----------------------------------------------------------------- tensores
def to_tensor_batch(x_uint8: np.ndarray | torch.Tensor, device) -> torch.Tensor:
    """(N,H,W,3) uint8 -> (N,3,H,W) float normalizado ImageNet, no device."""
    t = torch.as_tensor(x_uint8).to(device, non_blocking=True)
    # .contiguous(): sem isso o tensor fica em layout channels-last e, no GTX 1650,
    # o cuDNN cai num caminho ~12x mais lento.
    t = t.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (t - mean) / std


def gpu_augment(x: torch.Tensor) -> torch.Tensor:
    """Aumento de dados em lote, na GPU (x já normalizado, N,3,H,W).

    Fissuras não têm orientação preferencial -> flips e rotações de 90°.
    Iluminação varia em campo -> brilho/contraste por amostra.
    """
    n = x.shape[0]
    dev = x.device
    flip_h = torch.rand(n, device=dev) < 0.5
    x = torch.where(flip_h.view(-1, 1, 1, 1), x.flip(3), x)
    flip_v = torch.rand(n, device=dev) < 0.5
    x = torch.where(flip_v.view(-1, 1, 1, 1), x.flip(2), x)
    if torch.rand(1).item() < 0.5:
        x = x.transpose(2, 3)
    contrast = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.7, 1.3)
    bright = torch.empty(n, 1, 1, 1, device=dev).uniform_(-0.4, 0.4)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return (x - mean) * contrast + mean + bright


def field_augment(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Aumento de "condições de campo" (melhoria após a análise de robustez), na GPU.

    Recebe/retorna tensor normalizado ImageNet. Em cada amostra sorteia uma condição:
    pouca luz (ganho+gama), superexposição, sombra de borda suave, ruído gaussiano
    ou obstáculo opaco (elipse, até ~20% da área). O rótulo não muda: sombra e
    obstáculo em superfície SEM fissura ensinam o modelo a não confundir borda com fissura.
    Os parâmetros são sorteados de forma diferente das perturbações fixas de `perturb.py`
    usadas na avaliação (formas, intensidades e implementação distintas).
    """
    n, _, h, w = x.shape
    dev = x.device
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    z = (x * std + mean).clamp(0, 1)
    apply = torch.rand(n, device=dev) < p
    kind = torch.randint(0, 5, (n,), device=dev)
    u = lambda lo, hi: torch.empty(n, 1, 1, 1, device=dev).uniform_(lo, hi)  # noqa: E731

    # 0) pouca luz: ganho 0.2-0.7 e gama 1.0-2.2
    dark = (z ** u(1.0, 2.2)) * u(0.2, 0.7)
    # 1) superexposição
    bright = (z * u(1.3, 2.0) + u(0.0, 0.25)).clamp(0, 1)
    # 2) sombra: semiplano com borda suave em ângulo aleatório
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=dev), torch.linspace(-1, 1, w, device=dev), indexing="ij")
    ang = torch.rand(n, 1, 1, device=dev) * 2 * np.pi
    off = torch.empty(n, 1, 1, device=dev).uniform_(-0.6, 0.6)
    soft = torch.empty(n, 1, 1, device=dev).uniform_(0.01, 0.15)
    side = torch.sigmoid((torch.cos(ang) * xx + torch.sin(ang) * yy - off) / soft).unsqueeze(1)
    shadow = z * (1 - u(0.3, 0.75) * side)
    # 3) ruído
    noisy = (z + torch.randn_like(z) * u(0.03, 0.12)).clamp(0, 1)
    # 4) obstáculo: elipse opaca de cor escura/neutra
    cy, cx = torch.empty(n, 1, 1, device=dev).uniform_(-0.7, 0.7), torch.empty(n, 1, 1, device=dev).uniform_(-0.7, 0.7)
    ry, rx = torch.empty(n, 1, 1, device=dev).uniform_(0.15, 0.45), torch.empty(n, 1, 1, device=dev).uniform_(0.15, 0.45)
    ell = ((((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1).float().unsqueeze(1)
    color = torch.rand(n, 3, 1, 1, device=dev) * 0.5
    occl = z * (1 - ell) + color * ell

    out = z
    for k, alt in enumerate([dark, bright, shadow, noisy, occl]):
        sel = (apply & (kind == k)).view(-1, 1, 1, 1)
        out = torch.where(sel, alt, out)
    return (out - mean) / std


# -------------------------------------------------------------------- treino
def train_model(
    arch: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 4,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: str = "cuda",
    seed: int = 42,
    amp: bool = False,
    field_aug: bool = False,
    log=print,
):
    """Fine-tuning completo. `field_aug=True` adiciona o aumento de condições de campo. `amp=False` por padrão: nas GTX 16xx o fp16 gera NaN
    (ResNet50) e ainda é mais lento que fp32 (sem Tensor Cores)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    amp = amp and device == "cuda"
    model = build_model(arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(x_train) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long)
    history, best = [], {"val_loss": 1e9}
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        perm = np.random.permutation(len(x_train))
        tot_loss, n_seen = 0.0, 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            xb = to_tensor_batch(x_train[idx], device)
            if field_aug:
                xb = field_augment(xb)
            xb = gpu_augment(xb)
            yb = y_train_t[idx].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                loss = F.cross_entropy(model(xb), yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"perda não finita na época {ep}, lote {i // batch_size}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot_loss += loss.item() * len(idx)
            n_seen += len(idx)
        logits_val = predict_logits(model, x_val, device)
        val_loss = F.cross_entropy(torch.as_tensor(logits_val), torch.as_tensor(y_val, dtype=torch.long)).item()
        p_val = torch.softmax(torch.as_tensor(logits_val), 1)[:, 1].numpy()
        val_f1 = skm.f1_score(y_val, p_val >= 0.5)
        rec = {
            "epoch": ep,
            "train_loss": tot_loss / n_seen,
            "val_loss": val_loss,
            "val_f1@0.5": val_f1,
            "val_acc": skm.accuracy_score(y_val, p_val >= 0.5),
            "seconds": time.perf_counter() - t0,
        }
        history.append(rec)
        log(
            f"[{arch}] época {ep}/{epochs}  loss={rec['train_loss']:.4f}  "
            f"val_loss={val_loss:.4f}  val_F1={val_f1:.4f}  ({rec['seconds']:.0f}s)"
        )
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": ep, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    return model, history, best["epoch"]


@torch.no_grad()
def predict_logits(model: nn.Module, x: np.ndarray, device: str = "cuda", batch_size: int = 128) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        xb = to_tensor_batch(x[i : i + batch_size], device)
        out.append(model(xb).float().cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------- calibração
def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017): ajusta T na validação."""
    lg = torch.as_tensor(logits, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def probs_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return torch.softmax(torch.as_tensor(logits) / temperature, 1)[:, 1].numpy()


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = np.where(p >= 0.5, p, 1 - p)
    pred = (p >= 0.5).astype(int)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


# ------------------------------------------------------------------ limiares
def choose_thresholds(p_val: np.ndarray, y_val: np.ndarray, min_recall: float = 0.99) -> dict:
    """Limiar de triagem: maior limiar com recall >= min_recall na validação."""
    prec, rec, thr = skm.precision_recall_curve(y_val, p_val)
    prec, rec = prec[:-1], rec[:-1]
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    ok = np.where(rec >= min_recall)[0]
    t_triage = float(thr[ok[-1]]) if len(ok) else 0.5
    return {"triagem": t_triage, "f1_max": float(thr[np.argmax(f1)]), "padrao": 0.5}


def classification_report(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = skm.confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "limiar": threshold,
        "acuracia": skm.accuracy_score(y, pred),
        "precisao": skm.precision_score(y, pred, zero_division=0),
        "recall": skm.recall_score(y, pred),
        "f1": skm.f1_score(y, pred),
        "especificidade": tn / (tn + fp),
        "roc_auc": skm.roc_auc_score(y, p),
        "pr_auc": skm.average_precision_score(y, p),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def train_or_load(name: str, ckpt_dir: str | Path, x_train, y_train, x_val, y_val, force: bool = False, log=print, arch: str | None = None, **cfg):
    """Treina e salva checkpoint + histórico; se já existir (e não `force`), carrega.

    Permite retomar o notebook sem retreinar quando a máquina está ocupada.
    """
    arch = arch or name
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    f_w, f_h = ckpt_dir / f"{name}.pt", ckpt_dir / f"{name}_hist.json"
    if f_w.exists() and f_h.exists() and not force:
        info = json.loads(f_h.read_text(encoding="utf-8"))
        model = build_model(arch, pretrained=False).to(cfg.get("device", "cuda"))
        model.load_state_dict(torch.load(f_w, map_location="cpu", weights_only=True))
        log(f"[{name}] checkpoint carregado ({f_w.name}, treinado em {info['seconds'] / 60:.1f} min)")
        return model, info["history"], info["best_epoch"], info["seconds"]
    t0 = time.time()
    model, hist, best_ep = train_model(arch, x_train, y_train, x_val, y_val, log=log, **cfg)
    secs = time.time() - t0
    torch.save(model.state_dict(), f_w)
    f_h.write_text(json.dumps({"history": hist, "best_epoch": best_ep, "seconds": secs, "config": {k: v for k, v in cfg.items() if k != "device"}}, indent=1), encoding="utf-8")
    return model, hist, best_ep, secs


# ----------------------------------------------------------------- exportação
def save_bundle(model: nn.Module, name: str, out_dir: str | Path, meta: dict, arch: str | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / f"{name}.pt")
    cfg = {"arch": arch or name, "name": name, "classes": CLASSES, "mean": IMAGENET_MEAN, "std": IMAGENET_STD, **meta}
    (out_dir / f"{name}.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir / f"{name}.pt"
