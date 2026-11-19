
# Davi Gravina – RM 565619
# Henry Andrade Browne – RM 562089
# Luiz Felipe dos Santos Tragi – RM 565009

# Youtube = https://youtu.be/kHIGilqiWgI


# CitizenTech · Visão Computacional para Manutenção Urbana

Módulo do protótipo **CitizenTech** que faz a **triagem automática de fissuras em concreto**
(fachadas, muros, pontes, calçadas) em fotos e vídeos, mostrando **onde** está o possível dano e a
**confiança** do modelo, para a prefeitura priorizar vistorias.

> ⚠️ É uma ferramenta de **triagem**. A presença de fissura numa imagem não determina gravidade nem
> segurança estrutural; isso exige vistoria de um profissional habilitado.

## Entregas

| Entrega | Onde |
|---|---|
| Notebook (dados → treino → avaliação → vídeo → desempenho) | [`notebooks/CitizenTech_Fissuras_VC.ipynb`](notebooks/CitizenTech_Fissuras_VC.ipynb) |
| Código do módulo | [`src/citizentech_vc/`](src/citizentech_vc) |
| Funcionalidade no CitizenTech | integrada ao protótipo **Zela** ([`integracao_zela/`](integracao_zela)): foto no *Novo relato*, fila da prefeitura e aba *Monitoramento por câmera* (código de visão em `integracao_zela/backend/app/vision/`) |
| App de laboratório (modelo isolado) | [`app/app.py`](app/app.py) |
| Vídeo processado (monitoramento) | [`demo/monitoramento_fissuras.mp4`](demo/monitoramento_fissuras.mp4) |
| Relatório | [`relatorio/Relatorio_CitizenTech_VC.pdf`](relatorio/Relatorio_CitizenTech_VC.pdf) (fonte em `.md`) |
| Vídeo de demonstração | YouTube (link no topo deste README) |
| Métricas, figuras e CSVs | [`reports/`](reports) |

## Acessos

| O quê | Endereço | Como acessar |
|---|---|---|
| **Zela (CitizenTech) com análise de fotos** | http://localhost:8501 | sobe com os 2 comandos abaixo |
| Área do cidadão | http://localhost:8501 → *🙋 Sou cidadão* | aberta: *Novo relato* aceita foto |
| Área da prefeitura | http://localhost:8501 → *🏛️ Equipe da prefeitura* | **código de acesso** = `MANAGER_TOKEN` do `integracao_zela/backend/.env` (se o `.env` foi criado a partir do `.env.example`, o código é `troque-este-token`) |
| API do Zela | http://localhost:8000 | `/health` mostra `vision_ready: true` quando a visão está ativa |
| Documentação da API | http://localhost:8000/docs | rotas de visão em `/api/v1/vision/*` (as de vídeo e sequência pedem o header `X-Manager-Token`) |
| App de laboratório (modelo isolado) | http://localhost:8502 | `F5` no VS Code ou `streamlit run app/app.py` |
| Deploy público do Zela | https://gs-2026-2-frontend.vercel.app | **sem** análise de fotos: a Vercel não roda PyTorch (a API responde `vision_unavailable`) |

Subir o Zela localmente (Python 3.12+, dentro da pasta `integracao_zela`):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt -r frontend/requirements.txt
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r backend/requirements-vision.txt
copy backend\.env.example backend\.env
```

(Sem GPU NVIDIA, troque `cu128` por `cpu`.) Depois, um terminal para cada:

```bash
cd backend && uvicorn app.main:app --port 8000
```

```bash
cd frontend && streamlit run app.py
```

Fotos de exemplo para enviar: `integracao_zela/frontend/assets/exemplos/` (com e sem fissura, e uma fachada).
O modelo usado pelo Zela já vem em `integracao_zela/backend/models/vision/`.
Sem chave do Gemini, rode `set LLM_PROVIDER=mock` no terminal da API antes de subi-la (classificador de demonstração).

> O código de acesso da prefeitura **não** fica escrito neste README de propósito: ele é uma credencial e vale também
> para o deploy. Consulte o `.env` e troque o valor se ele aparecer em vídeo ou print.

## Como rodar

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Baixe a base *Concrete Crack Images for Classification* (Mendeley, DOI 10.17632/5y9wdsg2zt.2), extraia
para uma pasta com `Positive/` e `Negative/` e aponte a variável `CRACK_DATA` para ela:

```bash
set CRACK_DATA=C:/Users/<voce>/datasets/concrete_crack
```

Este pacote traz só o modelo final (`models/resnet18_campo.pt`) para caber no envio; os outros três são gerados de novo pelo treino. Treinar e avaliar (gera `models/`, `reports/` e `demo/`). O treino pode rodar antes, fora do notebook, com log em
tempo real. O notebook reaproveita os checkpoints de `models/ckpt/`; sem eles, treina dentro do próprio notebook:

```bash
python tools/train.py
python tools/build_notebook.py
python -m ipykernel install --user --name citizentech
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=citizentech notebooks/CitizenTech_Fissuras_VC.ipynb
python tools/make_examples.py
```

Abrir o app de laboratório (o modelo isolado, fora do Zela):

```bash
streamlit run app/app.py --server.port 8502
```

## Estrutura

```
src/citizentech_vc/
  data.py      amostragem, grupos por foto de origem, split sem vazamento, cache
  model.py     MobileNetV3-L / ResNet18 / ResNet50, treino, calibração, limiar, métricas
  infer.py     CrackDetector: janelas deslizantes, Grad-CAM, checagem de qualidade, desenho
  perturb.py   sombra, pouca luz, superexposição, desfoque, ruído, JPEG, obstáculo, baixa resolução
  video.py     monitoramento quadro a quadro com alerta por histerese, geração do vídeo de simulação
  bench.py     latência e FPS
app/app.py     interface CitizenTech (Streamlit)
tools/         gerador do notebook e dos exemplos
```

**Google Colab:** abra o `.ipynb` no Colab com GPU (T4). A primeira célula recria o pacote (o código está embutido
no notebook) e baixa a base sozinha; o notebook então treina os 4 modelos do zero.

O notebook é gerado a partir de `tools/build_notebook.py`. Edite esse arquivo e rode-o de novo para
regenerar o `.ipynb`.
