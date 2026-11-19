"""Gera relatorio/Relatorio_CitizenTech_VC.pdf a partir do .md (HTML + Edge headless).

    python tools/report_pdf.py
"""
import base64
import html
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "relatorio" / "Relatorio_CitizenTech_VC.md"
HTML_OUT = ROOT / "relatorio" / "Relatorio_CitizenTech_VC.html"
PDF_OUT = ROOT / "relatorio" / "Relatorio_CitizenTech_VC.pdf"
FIGS = ROOT / "reports" / "figs"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

# figuras inseridas no fim de cada seção (número da seção -> [(arquivo, legenda, altura máx. em mm)])
FIGURAS = {
    "1": [("03_vazamento.png", "Distância do teste ao vizinho mais próximo no treino: split por grupo (usado) × split aleatório.", 52)],
    "3": [("06_confusao_roc_pr.png", "Modelo final: matrizes de confusão (limiar de triagem e de F1 máx.) e curvas ROC / Precisão×Recall dos 4 modelos.", 48),
          ("14_fachada.png", "Fachada simulada com recortes de teste: verdade (acima) e regiões sinalizadas pelo CitizenTech (abaixo).", 72)],
    "4": [("11_gradcam.png", "Grad-CAM em acertos e erros. O falso positivo é uma fissura capilar real rotulada como 'sem fissura'.", 62),
          ("13_robustez.png", "Robustez no teste: ResNet18 × ResNet18 + campo (esq.) e recall/precisão do modelo final por condição (dir.).", 58)],
}

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 9.6pt; line-height: 1.42; color: #1d2433; }
h1 { font-size: 19pt; margin: 0 0 2pt; color: #12355b; }
h3.sub { font-size: 11.5pt; font-weight: 500; margin: 0 0 10pt; color: #44536b; }
h2 { font-size: 13pt; color: #12355b; border-bottom: 1.5pt solid #c9d6e8; padding-bottom: 2pt; margin: 14pt 0 6pt; break-after: avoid; }
p { margin: 4pt 0 6pt; text-align: justify; }
ul, ol { margin: 3pt 0 6pt 16pt; padding: 0; } li { margin: 2pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 8pt; font-size: 8.6pt; }
tr { break-inside: avoid; }
th { background: #12355b; color: #fff; font-weight: 600; text-align: left; padding: 3pt 5pt; }
td { border-bottom: .6pt solid #d9e1ec; padding: 2.5pt 5pt; vertical-align: top; }
tr:nth-child(even) td { background: #f4f7fb; }
blockquote { margin: 8pt 0; padding: 6pt 10pt; background: #fff6e5; border-left: 3pt solid #e8a33d; }
blockquote p { margin: 0; }
code { font-family: Consolas, monospace; font-size: 8.6pt; background: #eef2f7; padding: 0 2pt; border-radius: 2pt; }
hr { border: 0; border-top: .8pt solid #c9d6e8; margin: 10pt 0; }
figure { margin: 8pt 0 10pt; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; max-height: 92mm; }
figcaption { font-size: 8.2pt; color: #55627a; margin-top: 3pt; }
strong { color: #0f2a48; }
"""


def inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![*\w])\*([^*\n]+)\*(?![*\w])", r"<em>\1</em>", s)
    return s


def figure(name: str, caption: str, h_mm: int = 80) -> str:
    data = base64.b64encode((FIGS / name).read_bytes()).decode()
    return f'<figure><img style="max-height:{h_mm}mm" src="data:image/png;base64,{data}"><figcaption>{inline(caption)}</figcaption></figure>'


def convert(md: str) -> str:
    out, lines, i = [], md.splitlines(), 0
    section = None

    def close_section():
        for fig in FIGURAS.get(section, []):
            out.append(figure(*fig))

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1
            continue
        if s.startswith("## "):
            close_section()
            section = (re.match(r"##\s+(\d+)\.", s) or [None, None])[1]
            out.append(f"<h2>{inline(s[3:])}</h2>")
        elif s.startswith("### "):
            out.append(f'<h3 class="sub">{inline(s[4:])}</h3>')
        elif s.startswith("# "):
            out.append(f"<h1>{inline(s[2:])}</h1>")
        elif s == "---":
            out.append("<hr>")
        elif s.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip())
                i += 1
            out.append(f"<blockquote><p>{inline(' '.join(buf))}</p></blockquote>")
            continue
        elif s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, body = rows[0], [r for r in rows[1:] if not set("".join(r)) <= set("-: ")]
            t = "<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead><tbody>"
            t += "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body)
            out.append(t + "</tbody></table>")
            continue
        elif re.match(r"^(\*|-|\d+\.)\s", s):
            ordered = bool(re.match(r"^\d+\.", s))
            items = []
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(("|", "#", ">")):
                t = lines[i].strip()
                m = re.match(r"^(\*|-|\d+\.)\s+(.*)", t)
                if m and not lines[i].startswith("   "):
                    items.append(m.group(2))
                elif lines[i].startswith(" ") and items:  # continuação do item (inclui tabela recuada? não)
                    items[-1] += " " + t
                else:
                    break
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            continue
        else:
            buf = [s]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||>|\*\s|-\s|\d+\.\s|---)", lines[i].strip()):
                buf.append(lines[i].strip())
                i += 1
            out.append(f"<p>{inline(' '.join(buf))}</p>")
            continue
        i += 1
    close_section()
    return "\n".join(out)


body = convert(MD.read_text(encoding="utf-8"))
HTML_OUT.write_text(
    f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>Relatório CitizenTech VC</title>'
    f"<style>{CSS}</style></head><body>{body}</body></html>",
    encoding="utf-8",
)
subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={PDF_OUT}", HTML_OUT.as_uri()], check=True, capture_output=True, timeout=180)
HTML_OUT.unlink()
print("ok:", PDF_OUT, PDF_OUT.stat().st_size // 1024, "KB")
