"""Gera notebooks/CitizenTech_Fissuras_VC.ipynb (fonte do notebook em texto, fácil de revisar)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md("""
# CitizenTech · Visão Computacional para Manutenção Urbana
## Triagem automática de fissuras em superfícies de concreto

**Objetivo.** Sinalizar automaticamente imagens de fachadas, muros, pontes e calçadas que **merecem inspeção**,
mostrando onde está o possível dano e a confiança do modelo, para que a prefeitura priorize vistorias.

**Tarefa escolhida:** classificação binária (com / sem fissura) com **transferência de aprendizado**,
estendida para *localização aproximada* por janelas deslizantes + Grad-CAM e para **monitoramento em vídeo**.

**Base:** *Concrete Crack Images for Classification* (Özgenel, 2018, Mendeley Data, DOI 10.17632/5y9wdsg2zt.2):
40 000 recortes RGB 227×227 (20 000 com fissura, 20 000 sem), extraídos de 458 fotos de 4032×3024 de prédios do campus METU.

> ⚠️ **Limite de uso.** O sistema faz **triagem**: indica *onde olhar*. A presença de fissura numa imagem,
> isoladamente, **não** determina gravidade nem segurança estrutural — isso exige vistoria de profissional habilitado.

Roteiro: 1) dados → 2) split sem vazamento → 3) pré-processamento → 4) treino de 3 arquiteturas →
5) calibração e limiar → 6) métricas → 7) análise de erros e robustez → 8) localização → 9) vídeo → 10) desempenho.
""")

COLAB_MD = """
### ▶️ Rodando no Google Colab
Use **Ambiente de execução → Alterar o tipo → GPU (T4)**. A célula abaixo só age no Colab: recria o pacote
`citizentech_vc` (código embutido neste notebook), baixa a base do Mendeley (~240 MB) e a extrai em `/content`.
Localmente ela não faz nada e o notebook usa a pasta `src/` do projeto.
"""


def colab_setup_source() -> str:
    """Célula de bootstrap para o Colab, com o código de src/citizentech_vc embutido."""
    import json

    src = {p.name: p.read_text(encoding="utf-8") for p in sorted((ROOT / "src" / "citizentech_vc").glob("*.py"))}
    payload = json.dumps(src, ensure_ascii=False)
    assert "'''" not in payload
    return (
        "import sys, os, json, subprocess, shutil, urllib.request\n"
        "from pathlib import Path\n"
        "IN_COLAB = 'google.colab' in sys.modules\n"
        "if IN_COLAB:\n"
        "    PROJ = Path('/content/citizentech'); PKG = PROJ / 'src' / 'citizentech_vc'\n"
        "    PKG.mkdir(parents=True, exist_ok=True)\n"
        "    SRC = json.loads(r'''" + payload + "''')\n"
        "    for name, txt in SRC.items():\n"
        "        (PKG / name).write_text(txt, encoding='utf-8')\n"
        "    DATA_DIR = Path('/content/concrete_crack')\n"
        "    if not (DATA_DIR / 'Positive').exists():\n"
        "        rar = Path('/content/concrete.rar')\n"
        "        # Alternativa: se você já subiu o .rar para o Google Drive, ele é usado no lugar do download\n"
        "        drive_rar = [p for p in Path('/content/drive/MyDrive').glob('**/Concrete Crack Images for Classification.rar')] if Path('/content/drive/MyDrive').exists() else []\n"
        "        if drive_rar:\n"
        "            shutil.copy(drive_rar[0], rar); print('usando', drive_rar[0])\n"
        "        elif not rar.exists() or rar.stat().st_size < 241_000_000:\n"
        "            url = 'https://data.mendeley.com/public-files/datasets/5y9wdsg2zt/files/8a70d8a5-bce9-4291-bab9-b48cfb3e87c3/file_downloaded'\n"
        "            # O Mendeley responde 403 ao User-Agent padrão do urllib; com um de navegador funciona\n"
        "            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36'})\n"
        "            print('baixando a base (~240 MB)...')\n"
        "            with urllib.request.urlopen(req, timeout=600) as r, open(rar, 'wb') as fh:\n"
        "                shutil.copyfileobj(r, fh, 1 << 20)\n"
        "        if shutil.which('unrar') is None:\n"
        "            subprocess.run(['apt-get', 'install', '-y', '-qq', 'unrar'], check=True)\n"
        "        DATA_DIR.mkdir(exist_ok=True)\n"
        "        subprocess.run(['unrar', 'x', '-o+', '-inul', '/content/concrete.rar', str(DATA_DIR) + '/'], check=True)\n"
        "        rar.unlink(missing_ok=True)\n"
        "    try:\n"
        "        import imageio_ffmpeg  # noqa: F401\n"
        "    except ImportError:\n"
        "        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'imageio-ffmpeg'], check=True)\n"
        "    os.environ['CRACK_DATA'] = str(DATA_DIR)\n"
        "    os.chdir(PROJ)\n"
        "    print('Colab pronto:', PROJ, '| imagens:', len(list((DATA_DIR / 'Positive').glob('*.jpg'))), '+', len(list((DATA_DIR / 'Negative').glob('*.jpg'))))\n"
        "else:\n"
        "    print('Execução local: usando src/ do projeto')"
    )


md(COLAB_MD)
code(colab_setup_source())

code(r"""
import os, sys, json, time, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, matplotlib.pyplot as plt, cv2, torch
from sklearn import metrics as skm
from citizentech_vc import CLASSES, data, model as M, perturb, video, bench
from citizentech_vc.infer import CrackDetector, draw_analysis, overlay_cam, image_quality

DATA = Path(os.environ.get("CRACK_DATA", data.DEFAULT_ROOT))
CACHE = DATA.parent / "cache_citizentech"
FIGS = ROOT / "reports" / "figs"; FIGS.mkdir(parents=True, exist_ok=True)
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)
DEMO = ROOT / "demo"; DEMO.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from citizentech_vc.config import SEED, IMG, N_PER_CLASS
plt.rcParams.update({"figure.dpi": 110, "axes.grid": False, "font.size": 9})
torch.backends.cudnn.benchmark = True

HW = bench.hardware_info()
print(json.dumps(HW, indent=1, ensure_ascii=False)); print("dispositivo:", DEVICE, "| base:", DATA)
RESULTS = {"hardware": HW}
def savefig(name):
    plt.savefig(FIGS / f"{name}.png", bbox_inches="tight", dpi=130)
""")

md("## 1. Exame dos dados")
code(r"""
pos, neg = data.list_images(DATA)
print(f"Positive (com fissura): {len(pos)} | Negative (sem fissura): {len(neg)}")
shapes = pd.Series([cv2.imread(str(p)).shape for p in pos[:200:10] + neg[:200:10]]).value_counts()
print("Tamanhos (amostra):", dict(shapes))

rng = np.random.default_rng(0)
fig, ax = plt.subplots(2, 8, figsize=(14, 4))
for r, (lst, name) in enumerate([(pos, "com fissura"), (neg, "sem fissura")]):
    for c, i in enumerate(rng.choice(len(lst), 8, replace=False)):
        ax[r, c].imshow(cv2.imread(str(lst[i]))[..., ::-1]); ax[r, c].axis("off")
        ax[r, c].set_title(f"{name}\n{lst[i].name}", fontsize=7)
plt.suptitle("Exemplos da base: fissuras finas/grossas, manchas, textura rugosa, sombras"); plt.tight_layout(); savefig("01_exemplos"); plt.show()
""")

md("""
**Por que o split precisa de cuidado.** Os 40 mil recortes vêm de apenas 458 fotos: recortes da mesma foto compartilham
iluminação, cor e textura do concreto. Se recortes de uma mesma foto caem em treino e teste, o modelo "reconhece a parede"
e a métrica fica otimista. A base não informa a foto de origem, então medimos se a **numeração dos arquivos** preserva
a vizinhança: distância de aparência (cor Lab + bordas) entre arquivos separados por *k* posições.
""")
code(r"""
prof_pos = data.neighbour_distance_profile(pos[:3000])
prof_neg = data.neighbour_distance_profile(neg[:3000])
prof = pd.DataFrame({"Positive": prof_pos, "Negative": prof_neg}); prof.index.name = "distância na numeração"
display(prof.round(3))
ax = prof.plot(marker="o", logx=True, figsize=(6, 3)); ax.set_ylabel("dist. média de aparência (z)")
ax.set_title("Recortes vizinhos na numeração se parecem? (Positive: sim; Negative: não)"); savefig("02_vizinhanca"); plt.show()
RESULTS["vizinhanca"] = prof.round(3).to_dict()
""")
md("""
**Leitura:** (a pasta *Positive* tem duas sequências de numeração, `00001…10621` e `10000_1…19378_1`; ordenamos por sequência e número.) Nos *Positive* a distância cresce com o salto na numeração → arquivos próximos vêm da mesma foto
(agrupamos em **blocos contíguos de 100**). Nos *Negative* a distância é constante → a numeração foi embaralhada;
agrupamos por **aparência** (K-Means em cor/textura, 150 clusters), de modo que recortes da mesma superfície fiquem juntos.
O split 70/15/15 é feito **por grupo**, estratificado por classe.
""")

md("## 2. Amostra e split treino / validação / teste sem vazamento")
code(r"""
# N_PER_CLASS = 6000 (config.py): amostra de 12 mil imagens para caber no prazo de ~4h; a base completa tem 40 mil
t0 = time.time()
df = data.build_sample(DATA, n_per_class=N_PER_CLASS, seed=SEED)
df = data.split_by_group(df, seed=SEED)
print(f"amostra montada em {time.time()-t0:.0f}s")
tab = pd.crosstab(df.split, df.label.map(dict(enumerate(CLASSES))), margins=True)
display(tab)
print("grupos por conjunto:", df.groupby("split").group.nunique().to_dict())
print("grupos presentes em >1 conjunto:", int((df.groupby("group").split.nunique() > 1).sum()))
df.to_csv(ROOT / "reports" / "split.csv", index=False)
RESULTS["split"] = {s: {"n": int((df.split == s).sum()), "grupos": int(df[df.split == s].group.nunique())} for s in ["train", "val", "test"]}
""")

md("""
**Verificação do vazamento.** Para cada imagem de teste, medimos a distância até a imagem de treino mais parecida.
Num split aleatório (o "errado") o teste tem "gêmeos" no treino; no split por grupo, não.
""")
code(r"""
from concurrent.futures import ThreadPoolExecutor
from sklearn.neighbors import NearestNeighbors
with ThreadPoolExecutor(8) as ex:
    F = np.array(list(ex.map(lambda p: data.appearance_features(Path(p)), df.path)))
F = (F - F.mean(0)) / F.std(0)

def nn_dist(train_mask, test_mask):
    nn = NearestNeighbors(n_neighbors=1).fit(F[train_mask]); return nn.kneighbors(F[test_mask])[0][:, 0]
rand = np.random.default_rng(SEED).permutation(len(df)) < 0.7 * len(df)
d_group = nn_dist((df.split == "train").values, (df.split == "test").values)
d_rand = nn_dist(rand, ~rand)
print(f"dist. mediana teste→treino  | split por grupo: {np.median(d_group):.3f} | split aleatório: {np.median(d_rand):.3f}")
plt.figure(figsize=(6, 3)); plt.hist(d_rand, 60, alpha=.6, label="split aleatório"); plt.hist(d_group, 60, alpha=.6, label="split por grupo (usado)")
plt.xlabel("distância ao vizinho mais próximo no treino"); plt.legend(); plt.title("Teste por grupo fica mais 'longe' do treino"); savefig("03_vazamento"); plt.show()
RESULTS["vazamento"] = {"mediana_grupo": float(np.median(d_group)), "mediana_aleatorio": float(np.median(d_rand))}
""")

md("""
## 3. Pré-processamento
* **Redimensionamento** 227→224 (INTER_AREA, sem distorção: as imagens já são quadradas) — entrada nativa das redes ImageNet.
* **Normalização** pela média/desvio do ImageNet, porque usamos pesos pré-treinados nele.
* **Aumento de dados** (só treino, na GPU): flips H/V e transposição (fissura não tem orientação), brilho ±0,4 e contraste 0,7–1,3 (iluminação de campo).
* Imagens decodificadas uma vez e guardadas em cache `uint8` (≈1,8 GB em RAM) para o treino não ficar preso em leitura de JPEG.
""")
code(r"""
t0 = time.time()
X = {s: data.load_images(df[df.split == s].path, IMG, cache_dir=CACHE) for s in ["train", "val", "test"]}
Y = {s: df[df.split == s].label.to_numpy() for s in ["train", "val", "test"]}
P = {s: df[df.split == s].path.tolist() for s in ["train", "val", "test"]}
print({s: X[s].shape for s in X}, f"| {time.time()-t0:.0f}s")

xb = M.to_tensor_batch(X["train"][:6], DEVICE); xa = M.gpu_augment(xb)
std = torch.tensor(M.IMAGENET_STD, device=DEVICE).view(1,3,1,1); mean = torch.tensor(M.IMAGENET_MEAN, device=DEVICE).view(1,3,1,1)
show = lambda t: (t * std + mean).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
xc = M.gpu_augment(M.field_augment(xb, p=1.0))
fig, ax = plt.subplots(3, 6, figsize=(12, 6))
for i in range(6):
    for r, t in enumerate([xb, xa, xc]): ax[r, i].imshow(show(t)[i]); ax[r, i].axis("off")
ax[0, 0].set_title("original", loc="left"); ax[1, 0].set_title("aumento padrão", loc="left"); ax[2, 0].set_title("+ condições de campo (só no modelo *_campo)", loc="left")
savefig("04_aumento"); plt.show()
""")

md("""
## 4. Modelos — transferência de aprendizado
Comparamos três CNNs pré-treinadas no ImageNet, com *fine-tuning* completo (AdamW, OneCycle, fp32).

> **Nota de engenharia (GTX 1650).** Testamos precisão mista fp16: nas GTX 16xx (sem Tensor Cores) ela foi **mais lenta** que fp32 e gerou **NaN** no ResNet50, e o EfficientNet-B0 treinou a apenas ~24 img/s. Por isso treinamos e servimos em fp32 e trocamos o EfficientNet-B0 pelo ResNet18. Também foi preciso tornar o tensor de entrada contíguo após o `permute` (layout channels-last deixava o cuDNN ~12× mais lento).

| Modelo | Por que testar |
|---|---|
| **MobileNetV3-Large** | feita para dispositivos móveis/borda; candidata a tempo real (câmera de viatura, celular do fiscal) |
| **ResNet18** | ResNet rasa: meio-termo entre custo e capacidade |
| **ResNet50** | referência clássica (a mesma usada em aula com Keras), mais pesada |
| **ResNet18 + campo** | *melhoria*: a mesma ResNet18 treinada também com **aumento de condições de campo** (pouca luz, superexposição, sombra, ruído, obstáculo), criada depois de a análise de robustez da 1ª rodada mostrar quedas fortes nessas condições |

A seleção do modelo usa **somente a validação**; o teste é usado uma única vez no final.
""")
code(r"""
from citizentech_vc.config import CONFIGS   # mesma configuração de tools/train.py
FORCE_TRAIN = False   # True = retreina mesmo havendo checkpoint em models/ckpt/ (gerado por tools/train.py ou por este notebook)
print(json.dumps(CONFIGS, indent=1))
models, hist, train_time = {}, {}, {}
for arch, cfg in CONFIGS.items():
    models[arch], hist[arch], best_ep, train_time[arch] = M.train_or_load(
        arch, MODELS / "ckpt", X["train"], Y["train"], X["val"], Y["val"], force=FORCE_TRAIN, device=DEVICE, seed=SEED, **cfg)
    print(f"  -> melhor época {best_ep}, treino em {train_time[arch]/60:.1f} min\n")
    torch.cuda.empty_cache()
""")
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
for arch, h in hist.items():
    h = pd.DataFrame(h)
    ax[0].plot(h.epoch, h.train_loss, "--", label=f"{arch} treino"); ax[0].plot(h.epoch, h.val_loss, "-o", label=f"{arch} val")
    ax[1].plot(h.epoch, h["val_f1@0.5"], "-o", label=arch)
ax[0].set_title("Perda"); ax[0].set_xlabel("época"); ax[0].legend(fontsize=7); ax[1].set_title("F1 na validação"); ax[1].set_xlabel("época"); ax[1].legend(fontsize=7)
savefig("05_curvas"); plt.show()
""")

md("""
## 5. Calibração da confiança e escolha do limiar
* **Confiança calibrada.** Redes modernas tendem a ser confiantes demais. Ajustamos uma *temperatura* T na validação
  (temperature scaling): a probabilidade exibida no CitizenTech passa a significar algo próximo de "x% dos casos com essa confiança são fissura". Medimos pelo **ECE** (erro de calibração esperado).
* **Limiar de triagem.** Num serviço de manutenção, **deixar passar uma fissura (FN) custa mais** do que mandar um fiscal
  olhar uma parede sã (FP). Escolhemos na validação o **maior limiar com recall ≥ 99%**; também reportamos o limiar que maximiza F1 e o 0,5.
""")
code(r"""
calib, thr, logits = {}, {}, {}
for arch, m in models.items():
    logits[arch] = {s: M.predict_logits(m, X[s], DEVICE) for s in ["val", "test"]}
    T = M.fit_temperature(logits[arch]["val"], Y["val"])
    p_val_raw = M.probs_from_logits(logits[arch]["val"]); p_val = M.probs_from_logits(logits[arch]["val"], T)
    calib[arch] = {"T": T, "ece_val_antes": M.expected_calibration_error(p_val_raw, Y["val"]), "ece_val_depois": M.expected_calibration_error(p_val, Y["val"])}
    thr[arch] = M.choose_thresholds(p_val, Y["val"], min_recall=0.99)
display(pd.DataFrame(calib).T.round(4)); display(pd.DataFrame(thr).T.round(4))
""")

md("## 6. Métricas — validação (seleção) e teste (resultado final)")
code(r"""
rows = []
for arch in models:
    T = calib[arch]["T"]
    for split in ["val", "test"]:
        p = M.probs_from_logits(logits[arch][split], T)
        for name, t in thr[arch].items():
            r = M.classification_report(Y[split], p, t); r.update(modelo=arch, conjunto=split, criterio=name); rows.append(r)
met = pd.DataFrame(rows)
cols = ["modelo", "conjunto", "criterio", "limiar", "acuracia", "precisao", "recall", "f1", "especificidade", "roc_auc", "pr_auc", "fp", "fn"]
display(met[cols].round(4))
met[cols].to_csv(ROOT / "reports" / "metricas.csv", index=False)
""")

code(r"""
# Seleção usando SÓ a validação:
#  1) qualidade em imagem limpa: F1 no limiar de triagem a até 0,5 p.p. do melhor;
#  2) entre esses, maior F1 médio sob as perturbações de campo (aplicadas à VALIDAÇÃO);
#  3) empate (< 0,5 p.p.) -> menor latência.
lat_quick = {a: bench.bench_model(a, DEVICE, batch=1, half=False, n_iter=50, model=models[a])["lat_img_ms"] for a in models}
val_tri = met[(met.conjunto == "val") & (met.criterio == "triagem")].set_index("modelo")
Xval_pert = {name: perturb.apply_all(X["val"], name, seed=SEED + 1) for name in perturb.PERTURBATIONS if name != "original"}
rob_val = {a: np.mean([M.classification_report(Y["val"], M.probs_from_logits(M.predict_logits(models[a], xp, DEVICE), calib[a]["T"]), thr[a]["triagem"])["f1"]
                       for xp in Xval_pert.values()]) for a in models}
del Xval_pert
sel_tab = pd.DataFrame({"F1 val limpa": val_tri.f1, "F1 val médio c/ perturbações": pd.Series(rob_val), "latência lote 1 (ms)": pd.Series(lat_quick)})
display(sel_tab.round(4))
cands = [a for a in models if val_tri.loc[a, "f1"] >= val_tri.f1.max() - 0.005]
best_rob = max(rob_val[a] for a in cands)
cands = [a for a in cands if rob_val[a] >= best_rob - 0.005]
SELECTED = min(cands, key=lambda a: lat_quick[a])
print("==> modelo selecionado:", SELECTED)
RESULTS["selecao"] = sel_tab.round(4).to_dict(orient="index")
for name_, m in models.items():
    M.save_bundle(m, name_, MODELS, {"img_size": IMG, "temperature": calib[name_]["T"], "thresholds": thr[name_], "field_aug": bool(CONFIGS[name_].get("field_aug", False)),
                                     "params_M": round(M.count_params(m), 2), "train_minutes": round(train_time[name_] / 60, 2)}, arch=CONFIGS[name_]["arch"])
(MODELS / "selected.json").write_text(json.dumps({"arch": SELECTED}), encoding="utf-8")
RESULTS["selecionado"] = SELECTED
""")

code(r"""
T, t_tri = calib[SELECTED]["T"], thr[SELECTED]["triagem"]
p_test = M.probs_from_logits(logits[SELECTED]["test"], T); y_test = Y["test"]
fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
for i, (name, t) in enumerate([("triagem", t_tri), ("F1 máx.", thr[SELECTED]["f1_max"])]):
    cm = skm.confusion_matrix(y_test, p_test >= t)
    skm.ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax[i], colorbar=False, cmap="Blues")
    ax[i].set_title(f"{SELECTED}\nlimiar {name} = {t:.3f}")
for arch in models:
    p = M.probs_from_logits(logits[arch]["test"], calib[arch]["T"])
    fpr, tpr, _ = skm.roc_curve(y_test, p); pr, rc, _ = skm.precision_recall_curve(y_test, p)
    ax[2].plot(fpr, tpr, label=f"{arch} AUC={skm.auc(fpr, tpr):.4f}"); ax[3].plot(rc, pr, label=arch)
ax[2].set_xlim(0, .1); ax[2].set_ylim(.9, 1.001); ax[2].set_title("ROC (zoom)"); ax[2].set_xlabel("FPR"); ax[2].legend(fontsize=7)
ax[3].set_xlim(.9, 1.001); ax[3].set_ylim(.9, 1.001); ax[3].set_title("Precisão × Recall (zoom)"); ax[3].set_xlabel("recall")
plt.tight_layout(); savefig("06_confusao_roc_pr"); plt.show()

# diagrama de confiabilidade
p_raw = M.probs_from_logits(logits[SELECTED]["test"])
fig, ax = plt.subplots(figsize=(4, 3.6))
for pp, lab in [(p_raw, "sem calibração"), (p_test, f"calibrado (T={T:.2f})")]:
    pt, pm = __import__("sklearn.calibration", fromlist=["x"]).calibration_curve(y_test, pp, n_bins=10, strategy="quantile")
    ax.plot(pm, pt, "o-", label=f"{lab}  ECE={M.expected_calibration_error(pp, y_test):.4f}")
ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set_xlabel("confiança prevista"); ax.set_ylabel("fração real de fissuras"); ax.legend(fontsize=7)
ax.set_title("Confiabilidade (teste)"); savefig("07_calibracao"); plt.show()
""")

md("## 7. Análise de acertos e erros")
code(r"""
pred = p_test >= t_tri
idx_fp = np.where(pred & (y_test == 0))[0]; idx_fn = np.where(~pred & (y_test == 1))[0]
idx_tp = np.where(pred & (y_test == 1))[0]; idx_tn = np.where(~pred & (y_test == 0))[0]
print(f"FP={len(idx_fp)}  FN={len(idx_fn)}  (de {len(y_test)} imagens de teste)")

def grid(idxs, title, name, n=8):
    idxs = sorted(idxs, key=lambda i: -abs(p_test[i] - t_tri))[:n]
    if not len(idxs): print("nenhum:", title); return
    fig, ax = plt.subplots(1, n, figsize=(2 * n, 2.4))
    for a in ax: a.axis("off")
    for a, i in zip(ax, idxs):
        a.imshow(X["test"][i]); a.set_title(f"p={p_test[i]:.3f}\n{Path(P['test'][i]).parent.name[:3]}/{Path(P['test'][i]).name}", fontsize=7)
    plt.suptitle(title); plt.tight_layout(); savefig(name); plt.show()
grid(idx_fp, "Falsos positivos (sem fissura, mas sinalizada)", "08_fp")
grid(idx_fn, "Falsos negativos (fissura não sinalizada)", "09_fn")
grid(idx_tp, "Acertos: fissuras sinalizadas", "10_tp", n=6)
""")

code(r"""
det = CrackDetector(SELECTED, MODELS, device=DEVICE)
sel = list(idx_tp[:3]) + list(idx_fp[:3]) + list(idx_fn[:2])
labels = ["acerto"] * 3 + ["falso positivo"] * min(3, len(idx_fp)) + ["falso negativo"] * min(2, len(idx_fn))
fig, ax = plt.subplots(2, len(sel), figsize=(2.1 * len(sel), 4.4))
for j, (i, lab) in enumerate(zip(sel, labels)):
    img = X["test"][i]; cam = det.gradcam(img)
    ax[0, j].imshow(img); ax[0, j].set_title(f"{lab}\np={p_test[i]:.2f}", fontsize=7)
    ax[1, j].imshow(overlay_cam(img, cam)); ax[0, j].axis("off"); ax[1, j].axis("off")
plt.suptitle("Grad-CAM: onde o modelo 'olhou' para decidir 'fissura'"); plt.tight_layout(); savefig("11_gradcam"); plt.show()
""")

code(r"""
# Que condições visuais concentram os erros? (brilho, contraste, nitidez, textura)
def cond(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return {"brilho": g.mean(), "contraste": g.std(), "nitidez (var. Laplaciano)": cv2.Laplacian(g, cv2.CV_64F).var()}
C = pd.DataFrame([cond(x) for x in X["test"]]); C["grupo"] = np.select([pred & (y_test == 0), ~pred & (y_test == 1), y_test == 1], ["FP", "FN", "TP"], "TN")
display(C.groupby("grupo").median().round(1))
RESULTS["erros_condicoes"] = C.groupby("grupo").median().round(1).to_dict()
""")

md("""
### Robustez a interferências de campo
Aplicamos ao conjunto de teste perturbações que imitam o que uma câmera na rua enfrenta — **sombra** de poste/marquise,
**pouca luz**, **superexposição**, **desfoque**, **borrão de movimento**, **ruído**, **compressão JPEG**, **obstáculo** cobrindo
parte da superfície e **baixa resolução** — e medimos a queda de recall/precisão no limiar de triagem de cada modelo.
Comparamos os 4 modelos; o contraste mais importante é **ResNet18 × ResNet18 + campo** (mesma arquitetura, só muda o aumento de dados).
As perturbações daqui (`perturb.py`, sementes fixas, sobre o TESTE) são implementações diferentes das usadas no treino (`model.field_augment`),
mas da mesma família de condições — o ganho mostra robustez a condições *simuladas*; a validação final precisa de fotos reais de campo.
""")
code(r"""
rob_all = []
examples = {}
for cond in perturb.PERTURBATIONS:
    Xp = perturb.apply_all(X["test"], cond, seed=SEED)
    examples[cond] = Xp[idx_tp[0]]
    for a in models:
        pp = M.probs_from_logits(M.predict_logits(models[a], Xp, DEVICE), calib[a]["T"])
        r = M.classification_report(y_test, pp, thr[a]["triagem"]); r.update(condição=cond, modelo=a); rob_all.append(r)
    del Xp
rob_all = pd.DataFrame(rob_all)
rob_all[["modelo", "condição", "precisao", "recall", "f1", "fp", "fn"]].to_csv(ROOT / "reports" / "robustez.csv", index=False)
f1_tab = rob_all.pivot(index="condição", columns="modelo", values="f1").loc[list(perturb.PERTURBATIONS)]
f1_tab.loc["média (perturbadas)"] = f1_tab.drop("original").mean()
print("F1 no TESTE por condição (limiar de triagem de cada modelo):"); display(f1_tab.round(4))
rob = rob_all[rob_all.modelo == SELECTED].set_index("condição")[["precisao", "recall", "f1", "especificidade", "fp", "fn"]]
print(f"Detalhe do modelo selecionado ({SELECTED}):"); display(rob.round(4))
RESULTS["robustez"] = rob.round(4).to_dict(orient="index")
RESULTS["robustez_f1_modelos"] = f1_tab.round(4).to_dict(orient="index")

fig, ax = plt.subplots(1, len(examples), figsize=(1.9 * len(examples), 2.4))
for a, (k, v) in zip(ax, examples.items()): a.imshow(v); a.set_title(k, fontsize=7); a.axis("off")
plt.tight_layout(); savefig("12_perturbacoes"); plt.show()
cmp_cols = [c for c in ["resnet18", "resnet18_campo"] if c in f1_tab.columns] or list(f1_tab.columns)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
f1_tab.drop("média (perturbadas)")[cmp_cols].plot.barh(ax=ax[0]); ax[0].set_xlabel("F1 (teste)"); ax[0].set_title("Melhoria: aumento de condições de campo"); ax[0].legend(fontsize=7)
rob[["recall", "precisao"]].plot.barh(ax=ax[1]); ax[1].set_title(f"Selecionado: {SELECTED} (limiar {t_tri:.2f})"); ax[1].legend(fontsize=7)
plt.tight_layout(); savefig("13_robustez"); plt.show()
""")

md("""
## 8. Localização em imagens maiores (fachada)
O modelo foi treinado em recortes; numa foto inteira usamos **janelas deslizantes** (sobreposição de 25%), juntamos as
janelas acima do limiar em regiões e desenhamos caixas com a confiança máxima. Para avaliar, montamos uma "fachada" de
6×16 recortes **de teste** com trechos fissurados contínuos, sabendo onde está cada fissura.
""")
code(r"""
test_df = df[df.split == "test"]
mosaic, truth = video.build_mosaic(test_df[test_df.label == 1].path.tolist(), test_df[test_df.label == 0].path.tolist(), rows=6, cols=16, seed=SEED)
res = det.analyze(mosaic, tile=227, overlap=0.0)       # sem sobreposição -> 1 janela = 1 recorte (avaliação)
grid_pred = np.array([t.prob for t in res.tiles]).reshape(truth.shape) >= t_tri
tile_rep = M.classification_report(truth.ravel(), np.array([t.prob for t in res.tiles]), t_tri)
print(f"janelas: {res.n_tiles} | latência total {res.latency_ms:.0f} ms | precisão {tile_rep['precisao']:.3f} recall {tile_rep['recall']:.3f}")
res_vis = det.analyze(mosaic, tile=227, overlap=0.25)
vis = draw_analysis(mosaic, res_vis, t_tri)
fig, ax = plt.subplots(2, 1, figsize=(14, 8))
ax[0].imshow(mosaic); ax[0].set_title("Fachada simulada (recortes de teste)"); ax[0].axis("off")
for r, c in zip(*np.where(truth)): ax[0].add_patch(plt.Rectangle((c * 227, r * 227), 227, 227, fill=False, ec="lime", lw=1.2))
ax[1].imshow(vis); ax[1].set_title(f"Regiões sinalizadas para inspeção (limiar {t_tri:.2f}) — verde acima = verdade"); ax[1].axis("off")
plt.tight_layout(); savefig("14_fachada"); plt.show()
cv2.imwrite(str(DEMO / "fachada_teste.jpg"), mosaic[..., ::-1]); cv2.imwrite(str(DEMO / "fachada_resultado.jpg"), vis[..., ::-1])
RESULTS["fachada"] = {k: tile_rep[k] for k in ["precisao", "recall", "f1"]} | {"janelas": res.n_tiles, "latencia_ms": res.latency_ms}
""")

md("""
## 9. Monitoramento contínuo (vídeo)
Simulamos uma câmera percorrendo uma fachada maior (8×22 recortes de teste) em zigue-zague, com **variação de luz**, **sombra de poste
passando** e **tremor**. Cada quadro 1024×576 é dividido em janelas 227 px; o alerta usa média móvel + histerese
(liga após 3 quadros acima do limiar; desliga abaixo de 0,8×limiar) para não piscar com um quadro ruim.
""")
code(r"""
mosaic_v, truth_v = video.build_mosaic(test_df[test_df.label == 1].path.tolist(), test_df[test_df.label == 0].path.tolist(), rows=8, cols=22, seed=7)
frames_in = list(video.pan_frames(mosaic_v, 1024, 576, n_frames=360))
video.write_video(frames_in, DEMO / "entrada_camera.mp4", fps=24)
out_frames, recs = [], []
t0 = time.perf_counter()
for f, r in video.monitor(det, frames_in, threshold=t_tri, tile=227):
    out_frames.append(f); recs.append(r)
wall = time.perf_counter() - t0
video.write_video(out_frames, DEMO / "monitoramento_fissuras.mp4", fps=24)
R = pd.DataFrame([r.__dict__ for r in recs]); R.to_csv(ROOT / "reports" / "monitoramento_quadros.csv", index=False)
vid_stats = {"quadros": len(R), "resolucao": "1024x576", "janelas_por_quadro": det.analyze(frames_in[0], tile=227).n_tiles,
             "lat_modelo_media_ms": R.latency_ms.mean(), "lat_modelo_p95_ms": R.latency_ms.quantile(.95),
             "fps_pipeline_completo": len(R) / wall, "quadros_em_alerta": int(R.alerta.sum())}
print(json.dumps(vid_stats, indent=1)); RESULTS["video"] = vid_stats

fig, ax = plt.subplots(2, 1, figsize=(11, 4.4), sharex=True)
ax[0].plot(R.frame, R.prob_max, lw=.8, label="conf. máx. no quadro"); ax[0].plot(R.frame, R.score_ema, lw=1.4, label="média móvel")
ax[0].axhline(t_tri, color="r", ls="--", lw=.8, label="limiar"); ax[0].fill_between(R.frame, 0, 1, where=R.alerta, color="r", alpha=.12, label="alerta")
ax[0].legend(fontsize=7, ncol=4); ax[0].set_ylabel("confiança")
ax[1].plot(R.frame, R.latency_ms, lw=.8); ax[1].set_ylabel("latência (ms)"); ax[1].set_xlabel("quadro")
savefig("15_monitoramento"); plt.show()
fig, ax = plt.subplots(1, 3, figsize=(15, 3))
for a, k in zip(ax, [40, 170, 300]): a.imshow(out_frames[k]); a.axis("off"); a.set_title(f"quadro {k}")
plt.tight_layout(); savefig("16_quadros_video"); plt.show()
""")

md("""
## 10. Desempenho: latência e FPS
Medimos a latência **só do modelo** (tensor pronto, `cuda.synchronize`, 15 aquecimentos + 100 medições) em GPU fp32/fp16 e CPU,
para lote 1 (uma foto enviada pelo cidadão) e lote 32 (janelas de um quadro de vídeo), e a **troca qualidade × velocidade** ao reduzir a resolução de entrada.
""")
code(r"""
B = []
uniq = {}
for n_, cfg_ in CONFIGS.items():   # mesma arquitetura = mesma latência -> mede uma vez
    uniq.setdefault(cfg_["arch"], models[n_])
for arch, m in uniq.items():
    for dev, half in ([("cuda", False), ("cuda", True)] if DEVICE == "cuda" else []) + [("cpu", False)]:
        for bs in [1, 32]:
            mm = M.build_model(arch, pretrained=False); mm.load_state_dict(m.state_dict())
            r = bench.bench_model(arch, dev, batch=bs, half=half, model=mm, n_iter=100 if dev == "cuda" else (30 if bs == 1 else 5), warmup=15 if dev == "cuda" else 3)
            r["params_M"] = round(M.count_params(m), 2); B.append(r)
    torch.cuda.empty_cache()
B = pd.DataFrame(B); display(B.round(2)); B.to_csv(ROOT / "reports" / "benchmark.csv", index=False)
RESULTS["benchmark"] = B.round(3).to_dict(orient="records")
""")
code(r"""
# Qualidade x velocidade variando a resolução de entrada do modelo selecionado
RS = []
for res_ in [224, 192, 160, 128, 96]:
    Xr = np.stack([cv2.resize(x, (res_, res_), interpolation=cv2.INTER_AREA) for x in X["test"]])
    pr = M.probs_from_logits(M.predict_logits(models[SELECTED], Xr, DEVICE), T)
    rep = M.classification_report(y_test, pr, t_tri)
    lat = bench.bench_model(SELECTED, DEVICE, res=res_, batch=1, half=False, model=models[SELECTED], n_iter=100)
    RS.append({"resolucao": res_, "f1": rep["f1"], "recall": rep["recall"], "precisao": rep["precisao"], "lat_img_ms": lat["lat_img_ms"], "fps_lote1": lat["imgs_por_s"]})
RS = pd.DataFrame(RS); display(RS.round(4)); RS.to_csv(ROOT / "reports" / "resolucao.csv", index=False); RESULTS["resolucao"] = RS.round(4).to_dict(orient="records")

# Ponta a ponta de uma foto (leitura -> pré-processamento -> modelo -> probabilidade), como no app
img0 = X["test"][0].copy()
ts = []
for i in range(60):
    t0 = time.perf_counter(); det.predict_crops([img0]); torch.cuda.synchronize() if DEVICE == "cuda" else None; ts.append(time.perf_counter() - t0)
e2e = {"lat_media_ms": 1000 * np.mean(ts[10:]), "lat_p95_ms": 1000 * np.percentile(ts[10:], 95), "fps": 1 / np.mean(ts[10:])}
print("ponta a ponta, 1 imagem:", {k: round(v, 2) for k, v in e2e.items()}); RESULTS["ponta_a_ponta_1img"] = e2e

fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))
g = B[(B.lote == 1)]; lab = g.arquitetura + " " + g.dispositivo + " " + g.precisao
ax[0].barh(lab, g.lat_img_ms); ax[0].set_xscale("log"); ax[0].set_xlabel("ms por imagem (lote 1, log)"); ax[0].set_title("Latência por modelo/dispositivo")
ax[1].plot(RS.lat_img_ms, RS.f1, "o-")
for _, r in RS.iterrows(): ax[1].annotate(f"{int(r.resolucao)}px", (r.lat_img_ms, r.f1), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax[1].set_xlabel("ms por imagem"); ax[1].set_ylabel("F1 (teste)"); ax[1].set_title(f"Resolução: qualidade × velocidade ({SELECTED})")
plt.tight_layout(); savefig("17_desempenho"); plt.show()
""")

md("## 11. Resumo salvo para o relatório e o app")
code(r"""
final = met[(met.conjunto == "test")].set_index(["modelo", "criterio"])
RESULTS["teste"] = {f"{a}|{c}": final.loc[(a, c)][["limiar", "acuracia", "precisao", "recall", "f1", "especificidade", "roc_auc", "pr_auc", "fp", "fn", "tp", "tn"]].astype(float).round(4).to_dict()
                    for a, c in final.index}
RESULTS["validacao"] = {f"{a}|{c}": r[["limiar", "precisao", "recall", "f1"]].astype(float).round(4).to_dict()
                        for (a, c), r in met[met.conjunto == "val"].set_index(["modelo", "criterio"]).iterrows()}
RESULTS["calibracao"] = calib; RESULTS["limiares"] = thr
RESULTS["treino"] = {a: {"min": round(train_time[a] / 60, 2), "historico": hist[a]} for a in models}
(ROOT / "reports" / "metrics.json").write_text(json.dumps(RESULTS, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
sel = RESULTS["teste"][f"{SELECTED}|triagem"]
print(f"Modelo: {SELECTED} | limiar triagem {sel['limiar']:.3f} | precisão {sel['precisao']:.4f} | recall {sel['recall']:.4f} | F1 {sel['f1']:.4f}")
""")

import sys as _sys
if "--patch" in _sys.argv:
    # Insere/atualiza só a célula do Colab num notebook já executado (preserva as saídas)
    out = ROOT / "notebooks" / "CitizenTech_Fissuras_VC.ipynb"
    nb = nbf.read(out, as_version=4)
    nb.cells = [c for c in nb.cells if "IN_COLAB = 'google.colab'" not in c.source and "Rodando no Google Colab" not in c.source]
    nb.cells[1:1] = [nbf.v4.new_markdown_cell(COLAB_MD.strip()), nbf.v4.new_code_cell(colab_setup_source())]
    nbf.write(nb, out)
    print("patch ok:", out)
    raise SystemExit
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"name": "citizentech", "display_name": "CitizenTech (venv)", "language": "python"},
    "language_info": {"name": "python"},
}
out = ROOT / "notebooks" / "CitizenTech_Fissuras_VC.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("ok:", out)
