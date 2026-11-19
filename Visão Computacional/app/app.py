"""CitizenTech · Inspeção visual de fissuras (Streamlit).

Rodar:  streamlit run app/app.py
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from citizentech_vc.infer import CrackDetector, damage_regions, draw_analysis, heatmap_from_tiles, image_quality, overlay_cam  # noqa: E402
from citizentech_vc.video import monitor, read_video, write_video  # noqa: E402

MODELS = ROOT / "models"
DEMO = ROOT / "demo"
REPORTS = ROOT / "reports"
QUEUE = ROOT / "data" / "ocorrencias.csv"

st.set_page_config(page_title="CitizenTech · Inspeção Visual", page_icon="🏙️", layout="wide")
st.markdown(
    """<style>
    .badge{padding:.6rem 1rem;border-radius:.6rem;font-weight:700;font-size:1.15rem;display:inline-block}
    .alerta{background:#fde2e1;color:#9b1c1c;border:1px solid #f5b5b2}
    .ok{background:#ddf4e4;color:#14532d;border:1px solid #a7e3bd}
    .nota{font-size:.82rem;color:#666}
    </style>""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------- recursos
@st.cache_resource(show_spinner="Carregando modelo...")
def load_detector(arch: str) -> CrackDetector:
    det = CrackDetector(arch, MODELS)
    det.predict_crops([np.zeros((det.size, det.size, 3), np.uint8)] * 4)  # aquece cuDNN: 1ª análise sem latência inflada
    return det


def available_models() -> list[str]:
    return sorted(p.stem for p in MODELS.glob("*.pt"))


@st.cache_data
def load_metrics() -> dict:
    f = REPORTS / "metrics.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def read_upload(file) -> np.ndarray:
    return np.array(Image.open(file).convert("RGB"))


def priority(prob: float, frac: float, thr: float) -> str:
    if prob >= max(0.9, thr) and frac >= 0.15:
        return "Alta"
    if prob >= thr:
        return "Média"
    return "Baixa"


def save_occurrence(row: dict):
    QUEUE.parent.mkdir(exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(QUEUE, mode="a", header=not QUEUE.exists(), index=False, encoding="utf-8")


# ------------------------------------------------------------------ sidebar
if not available_models():
    st.error("Nenhum modelo em `models/`. Rode o notebook `notebooks/CitizenTech_Fissuras_VC.ipynb` primeiro.")
    st.stop()

sel_default = json.loads((MODELS / "selected.json").read_text())["arch"] if (MODELS / "selected.json").exists() else available_models()[0]
with st.sidebar:
    st.title("🏙️ CitizenTech")
    st.caption("Módulo de visão computacional · triagem de fissuras em concreto")
    arch = st.selectbox("Modelo", available_models(), index=available_models().index(sel_default))
    det = load_detector(arch)
    thr_opts = det.thresholds
    modo = st.radio(
        "Limiar de decisão",
        ["triagem", "f1_max", "padrao"],
        format_func=lambda k: {"triagem": f"Triagem (recall ≥ 99%) · {thr_opts['triagem']:.2f}",
                               "f1_max": f"Equilíbrio (F1 máx.) · {thr_opts['f1_max']:.2f}",
                               "padrao": "Padrão · 0.50"}[k],
    )
    thr = st.slider("Ajuste fino do limiar", 0.01, 0.99, float(thr_opts[modo]), 0.01)
    tile_auto = st.checkbox("Tamanho de janela automático", True)
    tile = None if tile_auto else st.slider("Janela (px)", 96, 640, 227, 1)
    dev = "GPU " + __import__("torch").cuda.get_device_name(0) if det.device == "cuda" else "CPU"
    st.caption(f"Dispositivo: **{dev}** · {'fp16' if det.half else 'fp32'} · entrada {det.size}px · T={det.temperature:.2f}")
    st.markdown(
        "<div class='nota'>⚠️ Ferramenta de <b>triagem</b>: indica onde inspecionar. "
        "A presença de fissura não determina gravidade nem segurança estrutural.</div>",
        unsafe_allow_html=True,
    )

tab_img, tab_vid, tab_fila, tab_modelo = st.tabs(["📷 Analisar imagem", "🎥 Monitoramento (vídeo/sequência)", "📋 Fila de inspeção", "ℹ️ Sobre o modelo"])

# ------------------------------------------------------------ analisar imagem
with tab_img:
    c_in, c_out = st.columns([1, 2], gap="large")
    with c_in:
        st.subheader("Nova ocorrência")
        src = st.radio("Imagem", ["Enviar foto", "Exemplo"], horizontal=True)
        img = None
        if src == "Enviar foto":
            up = st.file_uploader("Foto da superfície (JPG/PNG)", type=["jpg", "jpeg", "png", "webp"])
            if up:
                img = read_upload(up)
                name = up.name
        else:
            exemplos = sorted(DEMO.glob("exemplo_*.jpg")) + sorted(DEMO.glob("fachada_teste.jpg"))
            if exemplos:
                ex = st.selectbox("Exemplo", exemplos, format_func=lambda p: p.stem)
                img = read_upload(ex)
                name = ex.name
            else:
                st.info("Sem exemplos em demo/. Rode o notebook.")
        endereco = st.text_input("Endereço / referência", placeholder="Ex.: Viaduto do Chá, pilar 3")
        estrutura = st.selectbox("Tipo de estrutura", ["Fachada / muro", "Ponte / viaduto", "Calçada", "Escola / prédio público", "Outro"])
        if img is not None:
            st.image(img, caption=f"{name} · {img.shape[1]}×{img.shape[0]}px", width="stretch")

    with c_out:
        if img is None:
            st.info("Envie uma foto ou escolha um exemplo para analisar.")
        else:
            q = image_quality(img)
            res = det.analyze(img, tile=tile)
            heat = heatmap_from_tiles(res, img.shape)
            regions = damage_regions(heat, thr)
            alerta = res.prob_max >= thr
            pr = priority(res.prob_max, res.frac_tiles, thr)
            st.markdown(
                f"<span class='badge {'alerta' if alerta else 'ok'}'>"
                f"{'🔴 INSPEÇÃO RECOMENDADA' if alerta else '🟢 Sem indício de fissura'}</span>",
                unsafe_allow_html=True,
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Confiança (fissura)", f"{res.prob_max:.1%}")
            m2.metric("Área sinalizada", f"{res.frac_tiles:.0%}", help="Fração das janelas acima do limiar")
            m3.metric("Regiões", len(regions))
            m4.metric("Latência", f"{res.latency_ms:.0f} ms", help=f"{res.n_tiles} janela(s) de {res.tile_size}px")
            for a in q["alertas"]:
                st.warning(f"Qualidade da imagem: {a}. O resultado pode ser menos confiável; se possível, refaça a foto.")

            if res.n_tiles == 1:
                cam = det.gradcam(img)
                v1, v2 = st.columns(2)
                v1.image(img, caption="Imagem", width="stretch")
                v2.image(overlay_cam(img, cam), caption="Grad-CAM: regiões que pesaram para 'fissura'", width="stretch")
            else:
                st.image(draw_analysis(img, res, thr), caption=f"Regiões sinalizadas (limiar {thr:.2f}) · {res.n_tiles} janelas de {res.tile_size}px", width="stretch")
                if regions:
                    (x, y, w, h), p = regions[0]
                    crop = img[y : y + h, x : x + w]
                    if min(crop.shape[:2]) >= 32:
                        with st.expander("Detalhe da região mais provável (Grad-CAM)"):
                            g1, g2 = st.columns(2)
                            g1.image(crop, caption=f"região 1 · {p:.0%}", width="stretch")
                            g2.image(overlay_cam(crop, det.gradcam(crop)), caption="Grad-CAM", width="stretch")

            st.caption(f"Nitidez relativa (Laplaciano) {q['nitidez']:.1f} · brilho {q['brilho']:.0f} · contraste {q['contraste']:.0f}")
            if st.button("📌 Registrar na fila de inspeção", type="primary", disabled=not alerta):
                save_occurrence({
                    "data_hora": datetime.now().isoformat(timespec="seconds"), "endereco": endereco or "(sem endereço)",
                    "estrutura": estrutura, "arquivo": name, "confianca": round(res.prob_max, 4),
                    "area_sinalizada": round(res.frac_tiles, 4), "regioes": len(regions), "prioridade_triagem": pr,
                    "modelo": arch, "limiar": thr, "alertas_qualidade": "; ".join(q["alertas"]), "status": "Aguardando vistoria",
                })
                st.success(f"Ocorrência registrada com prioridade de triagem **{pr}**.")

# ------------------------------------------------------------- monitoramento
with tab_vid:
    st.subheader("Monitoramento urbano")
    st.caption("Processa um vídeo curto (câmera veicular, drone, celular) ou uma sequência de fotos, quadro a quadro.")
    modo_v = st.radio("Entrada", ["Vídeo de demonstração", "Enviar vídeo", "Sequência de imagens"], horizontal=True)
    every = st.slider("Analisar 1 a cada N quadros", 1, 10, 1, help="Pular quadros aumenta o FPS; o último resultado é mantido nos quadros pulados.")

    if modo_v == "Sequência de imagens":
        files = st.file_uploader("Fotos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if files and st.button("Processar sequência", type="primary"):
            rows, t0 = [], time.perf_counter()
            cols = st.columns(4)
            for i, f in enumerate(files):
                im = read_upload(f)
                r = det.analyze(im, tile=tile)
                rows.append({"arquivo": f.name, "confiança": r.prob_max, "área sinalizada": r.frac_tiles,
                             "decisão": "Inspecionar" if r.prob_max >= thr else "OK", "latência (ms)": r.latency_ms})
                cols[i % 4].image(draw_analysis(im, r, thr) if r.n_tiles > 1 else im, caption=f"{f.name} · {r.prob_max:.0%}", width="stretch")
            wall = time.perf_counter() - t0
            df = pd.DataFrame(rows)
            st.dataframe(df.style.format({"confiança": "{:.1%}", "área sinalizada": "{:.0%}", "latência (ms)": "{:.1f}"}), width="stretch")
            st.metric("Taxa de processamento", f"{len(files) / wall:.1f} imagens/s")
    else:
        vpath = None
        if modo_v == "Vídeo de demonstração":
            vpath = DEMO / "entrada_camera.mp4"
            if not vpath.exists():
                st.info("Rode o notebook para gerar demo/entrada_camera.mp4.")
                vpath = None
        else:
            upv = st.file_uploader("Vídeo (MP4/AVI/MOV)", type=["mp4", "avi", "mov", "mkv"])
            if upv:
                tmp = Path(tempfile.gettempdir()) / f"citizentech_{upv.name}"
                tmp.write_bytes(upv.read())
                vpath = tmp
        if vpath is not None:
            st.video(str(vpath))
            if st.button("▶️ Processar vídeo", type="primary"):
                frames, fps = read_video(vpath, max_side=1024, max_frames=900)
                prog = st.progress(0.0, "Processando...")
                live = st.empty()
                out, recs = [], []
                t0 = time.perf_counter()
                for f, r in monitor(det, frames, threshold=thr, tile=tile or (227 if modo_v == "Vídeo de demonstração" else None), every=every, video_fps=fps):
                    out.append(f)
                    recs.append(r.__dict__)
                    if r.frame % 15 == 0:
                        live.image(f, caption=f"quadro {r.frame}", width="stretch")
                        prog.progress((r.frame + 1) / len(frames), f"quadro {r.frame + 1}/{len(frames)}")
                wall = time.perf_counter() - t0
                prog.empty()
                live.empty()
                outp = Path(tempfile.gettempdir()) / "citizentech_resultado.mp4"
                write_video(out, outp, fps=fps)
                R = pd.DataFrame(recs)
                st.video(str(outp))
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Quadros", len(R))
                k2.metric("FPS (pipeline completo)", f"{len(R) / wall:.1f}")
                k3.metric("Latência média do modelo", f"{R.latency_ms.mean():.0f} ms")
                k4.metric("Quadros em alerta", f"{int(R.alerta.sum())} ({R.alerta.mean():.0%})")
                st.line_chart(R.set_index("frame")[["prob_max", "score_ema"]].rename(columns={"prob_max": "confiança no quadro", "score_ema": "média móvel"}))
                # eventos de alerta (trechos contínuos)
                ev, start = [], None
                for _, r in R.iterrows():
                    if r.alerta and start is None:
                        start = r
                    if not r.alerta and start is not None:
                        ev.append({"início (s)": start.t_video, "fim (s)": r.t_video, "conf. máx.": R[(R.frame >= start.frame) & (R.frame < r.frame)].prob_max.max()})
                        start = None
                if start is not None:
                    ev.append({"início (s)": start.t_video, "fim (s)": R.t_video.iloc[-1], "conf. máx.": R[R.frame >= start.frame].prob_max.max()})
                st.markdown(f"**Trechos para inspeção:** {len(ev)}")
                if ev:
                    st.dataframe(pd.DataFrame(ev).style.format({"início (s)": "{:.1f}", "fim (s)": "{:.1f}", "conf. máx.": "{:.1%}"}), width="stretch")
                st.download_button("Baixar vídeo processado", outp.read_bytes(), "citizentech_monitoramento.mp4", "video/mp4")

# --------------------------------------------------------------------- fila
with tab_fila:
    st.subheader("Fila de inspeção")
    if QUEUE.exists():
        q = pd.read_csv(QUEUE)
        ordem = {"Alta": 0, "Média": 1, "Baixa": 2}
        q = q.sort_values(["prioridade_triagem", "confianca"], key=lambda s: s.map(ordem) if s.name == "prioridade_triagem" else -s)
        a, b, c = st.columns(3)
        a.metric("Ocorrências", len(q))
        b.metric("Prioridade alta", int((q.prioridade_triagem == "Alta").sum()))
        c.metric("Aguardando vistoria", int((q.status == "Aguardando vistoria").sum()))
        st.dataframe(q, width="stretch", hide_index=True)
        st.download_button("Exportar CSV", q.to_csv(index=False).encode("utf-8"), "fila_inspecao.csv", "text/csv")
    else:
        st.info("Nenhuma ocorrência registrada ainda. Analise uma imagem e clique em **Registrar na fila de inspeção**.")
    st.caption("Prioridade de triagem = confiança × área sinalizada. Ela ordena vistorias; não mede gravidade do dano.")

# -------------------------------------------------------------------- modelo
with tab_modelo:
    mt = load_metrics()
    st.subheader(f"Modelo em uso: {arch}")
    if mt:
        t = mt.get("teste", {})
        nomes = {"triagem": "triagem (recall ≥ 99%)", "f1_max": "F1 máx.", "padrao": "padrão 0,5"}
        linhas = [{"modelo": k.split("|")[0], "critério": nomes.get(k.split("|")[1], k.split("|")[1]), **v} for k, v in t.items()]
        if linhas:
            st.markdown("**Métricas no conjunto de teste** (split por imagem de origem, sem vazamento)")
            st.dataframe(pd.DataFrame(linhas)[["modelo", "critério", "limiar", "precisao", "recall", "f1", "especificidade", "roc_auc", "fp", "fn"]]
                         .style.format({**{c: "{:.4f}" for c in ["limiar", "precisao", "recall", "f1", "especificidade", "roc_auc"]}, "fp": "{:.0f}", "fn": "{:.0f}"}), width="stretch", hide_index=True)
        hw = mt.get("hardware", {})
        st.caption(f"Equipamento: {hw.get('gpu', '-')} · {hw.get('cpu', '-')} · {hw.get('ram_gb', '-')} GB RAM")
        figs = REPORTS / "figs"
        for nome, legenda in [("06_confusao_roc_pr", "Matriz de confusão e curvas"), ("11_gradcam", "Grad-CAM"), ("13_robustez", "Robustez"), ("17_desempenho", "Desempenho")]:
            if (figs / f"{nome}.png").exists():
                st.image(str(figs / f"{nome}.png"), caption=legenda, width="stretch")
    else:
        st.info("Métricas aparecem aqui depois de rodar o notebook.")
