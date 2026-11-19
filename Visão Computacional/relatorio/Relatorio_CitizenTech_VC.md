# CitizenTech · Visão Computacional para Manutenção Urbana
### Relatório técnico: triagem de fissuras em superfícies de concreto

**Problema escolhido:** identificação de fissuras em concreto (classificação binária com/sem fissura), estendida
para **localização aproximada** em fotos grandes (janelas deslizantes + Grad-CAM) e **monitoramento em vídeo**.
**Base:** *Concrete Crack Images for Classification* (Özgenel, 2018, Mendeley, DOI 10.17632/5y9wdsg2zt.2):
40 000 recortes 227×227 (metade com fissura), extraídos de 458 fotos de prédios.
**Equipamento:** notebook Intel Core i5-10300H, 32 GB RAM, **GPU NVIDIA GTX 1650 (4 GB)**, Windows 11,
Python 3.14, PyTorch 2.11 (CUDA 12.8). Resolução de entrada do modelo: **224×224**.

> ⚠️ O módulo faz **triagem**: indica onde inspecionar. A presença de fissura numa imagem, isoladamente,
> não determina a gravidade do dano nem a segurança estrutural da edificação.

---

## 1. Dados e preparação

* **Amostra:** 12 079 imagens (6 079 com fissura, 6 000 sem), compatível com o prazo de 4 h.
* **Split sem vazamento (70/15/15):** a base não informa de que foto veio cada recorte. Medimos que, na pasta
  *Positive*, arquivos vizinhos na numeração são muito parecidos (distância de aparência 1,4 entre vizinhos contra
  3,3 a 1 000 posições), ou seja, vêm da mesma foto. Agrupamos em **blocos contíguos de 100 arquivos**. Na pasta
  *Negative* a numeração é aleatória, então agrupamos por **aparência** (K-Means em cor Lab + bordas, 150 grupos).
  Treino, validação e teste recebem **grupos inteiros**: 151, 31 e 30 grupos, nenhum compartilhado.
  Verificação: a distância mediana do teste ao vizinho mais próximo no treino é 0,54 no nosso split e 0,43 num
  split aleatório. O split aleatório colocaria "gêmeos" do teste no treino.
* **Pré-processamento:** redimensionamento 227→224 (INTER_AREA) e normalização ImageNet, por causa dos pesos
  pré-treinados. **Aumento de dados** na GPU: flips, transposição (fissura não tem orientação), brilho e contraste.
* Observação: a pasta *Positive* tem duas sequências de nomes (`00001.jpg` e `10000_1.jpg`). Um `int("19378_1")`
  ingênuo em Python vale 193 781 e quebraria os grupos, por isso os nomes são interpretados explicitamente.

## 2. Arquitetura e justificativa

Transferência de aprendizado com *fine-tuning* completo (AdamW, OneCycle, fp32) de três CNNs pré-treinadas no
ImageNet, mais uma variante criada a partir da análise de erros:

| Modelo | Parâmetros | Treino (GTX 1650) | Por quê |
|---|---|---|---|
| MobileNetV3-Large | 4,2 M | 10,4 min | feita para dispositivo móvel/borda |
| ResNet18 | 11,2 M | 4,0 min | meio-termo custo × capacidade |
| ResNet50 | 23,5 M | 21,5 min | referência clássica (vista em aula) |
| **ResNet18 + campo** | 11,2 M | 5,1 min | ResNet18 + **aumento de condições de campo** (pouca luz, superexposição, sombra, ruído, obstáculo) |

**Seleção, feita só na validação:** (1) F1 em imagem limpa a até 0,5 p.p. do melhor; (2) entre esses, o maior
F1 médio sob 9 perturbações de campo aplicadas à **validação**; (3) desempate pela latência. Resultado:

| Modelo | F1 val. (limpa) | F1 val. médio c/ perturbações | Latência lote 1 |
|---|---|---|---|
| MobileNetV3-L | 0,994 | 0,698 | 21,4 ms |
| ResNet18 | 0,995 | 0,817 | 8,0 ms |
| ResNet50 | 0,994 | 0,879 | 17,2 ms |
| **ResNet18 + campo** ✅ | 0,994 | **0,971** | 9,3 ms |

**Por que ResNet18 e não MobileNetV3?** Na GTX 1650 a MobileNetV3 é **mais lenta em lote 1** (20,7 ms contra
6,1 ms). As convoluções *depthwise* são limitadas pelo disparo de kernels, não por FLOPs, e ela foi a mais frágil a
ruído e pouca luz. A MobileNet só ganha em lote grande (870 img/s contra 647 img/s) e em CPU.

**Por que fp32?** Testamos precisão mista fp16. Nas GTX 16xx (sem Tensor Cores) ela foi **mais lenta** que fp32
(ResNet18: 12,5 ms contra 6,1 ms) e gerou **NaN** no ResNet50 (lote 32). Também foi necessário tornar o tensor
contíguo após o `permute` HWC→CHW: o layout *channels-last* deixava o cuDNN cerca de 12× mais lento nessa GPU.

**Confiança e limiar.** A confiança exibida é calibrada por *temperature scaling* na validação (T = 1,16; ECE na
validação de 0,0017 para 0,0016; o modelo já era bem calibrado). O **limiar de triagem é 0,94**, o maior limiar
com **recall ≥ 99% na validação**. Numa triagem urbana, deixar passar uma fissura custa mais que enviar um fiscal
a uma parede sã. O app permite trocar para o limiar de F1 máximo (0,14) ou ajustar manualmente.

## 3. Resultados (conjunto de teste: 1 709 imagens, 30 grupos nunca vistos)

| Modelo (limiar de triagem) | Precisão | Recall | F1 | Especificidade | ROC-AUC | FP | FN |
|---|---|---|---|---|---|---|---|
| MobileNetV3-L (0,95) | 1,0000 | 0,9977 | 0,9989 | 1,0000 | 1,0000 | 0 | 2 |
| ResNet18 (0,98) | 1,0000 | 0,9954 | 0,9977 | 1,0000 | 1,0000 | 0 | 4 |
| ResNet50 (0,95) | 1,0000 | 0,9932 | 0,9966 | 1,0000 | 1,0000 | 0 | 6 |
| **ResNet18 + campo (0,94)** | **0,9989** | **0,9989** | **0,9989** | 0,9988 | 1,0000 | **1** | **1** |

Matriz de confusão do modelo final: 830 VN, 1 FP, 1 FN, 877 VP. Figuras: `reports/figs/06_confusao_roc_pr.png`
e `07_calibracao.png`.

**Localização na fachada simulada** (6×16 recortes de teste, 22% fissurados): precisão, recall e F1 de 1,00
por janela, com 96 janelas analisadas em 2,6 s (`14_fachada.png`).

## 4. Análise de erros

* **Erros em imagem limpa (2 de 1 709).** O falso positivo (p = 0,96) é uma **fissura capilar real** numa imagem
  rotulada "sem fissura": o Grad-CAM acompanha exatamente a linha (`11_gradcam.png`). É um erro de rótulo da base,
  não do modelo. O falso negativo (p = 0,89) é uma fissura **fina e de baixo contraste**, logo abaixo do limiar.
  Nas medianas, os erros têm contraste menor (FN 9 contra 31 dos acertos) e brilho menor (143 contra 164).
* **Texturas.** Superfícies sem fissura, mas muito texturizadas (concreto rugoso, agregado aparente, manchas), concentram
  os alarmes falsos: a energia de bordas (variância do Laplaciano) dos falsos positivos tem mediana 124, contra 11 nas
  superfícies sãs acertadas, ou seja, **11× mais textura**. O modelo confunde poros e arestas de agregado com o início de
  uma fissura. Por isso o limiar de triagem é alto (0,94) e a decisão final é sempre da vistoria.
* **Interferências de campo** (F1 no teste, limiar de triagem):

| Condição | ResNet18 | ResNet18 + campo | Leitura |
|---|---|---|---|
| original | 0,998 | 0,999 | sem perda em imagem limpa |
| **pouca luz** | **0,230** | **0,997** | sem o aumento, foto escura = "sem fissura" (recall 13%) |
| **sombra** | 0,835 | 0,998 | borda de sombra virava "fissura" (327 FP) |
| **obstáculo** | 0,691 | 0,957 | bordas de objeto viravam "fissura" (783 FP → 60 FP) |
| ruído | 0,853 | 0,999 | |
| superexposição | 0,899 | 0,994 | |
| JPEG forte | 0,998 | 0,998 | |
| desfoque | 0,998 | 0,951 | ⬇ piorou |
| borrão de movimento | 0,992 | 0,864 | ⬇ piorou (recall 76%) |
| baixa resolução | 0,998 | 0,959 | ⬇ piorou |
| **média (perturbadas)** | **0,833** | **0,969** | |

  A melhoria eliminou as falhas graves (pouca luz, sombra, obstáculo, ruído), mas o modelo ficou **mais sensível
  a desfoque e baixa resolução**: aprendeu a associar fissura a borda nítida. Mitigação no app: uma **checagem de
  nitidez** (razão do Laplaciano antes e depois de um leve desfoque, calibrada em 400 recortes) avisa o usuário
  para refazer a foto. O próximo passo é incluir desfoque no aumento de campo.
  Ressalva: as perturbações de avaliação (`perturb.py`) têm implementação e sorteio diferentes das de treino, mas
  são da mesma família. Isso mede robustez a condições **simuladas**; a validação final exige fotos reais da cidade.

## 5. Desempenho em tempo real

Latência **só do modelo** (média de 100 medições após 15 de aquecimento, `cuda.synchronize`):

| Modelo | GPU fp32, lote 1 | GPU fp32, lote 32 (por imagem) | CPU, lote 1 |
|---|---|---|---|
| MobileNetV3-L | 20,7 ms (48 FPS) | 1,15 ms (871 img/s) | 52,9 ms (19 FPS) |
| **ResNet18** | **6,1 ms (163 FPS)** | 1,55 ms (647 img/s) | 61,8 ms (16 FPS) |
| ResNet50 | 14,4 ms (69 FPS) | 4,84 ms (207 img/s) | 194 ms (5 FPS) |

* **Uma foto enviada pelo cidadão** (pré-processamento + modelo + probabilidade): **8,7 ms em média**
  (p95 14,2 ms), cerca de 115 imagens/s.
* **Vídeo** 1024×576, 24 janelas de 227 px por quadro (janelas, modelo, mapa de calor e desenho):
  **6,6 a 8,5 FPS** de ponta a ponta, com o modelo levando 66 a 77 ms por quadro. Para monitoramento de viatura
  (5 a 10 FPS bastam) isso atende. Para mais FPS: analisar 1 a cada N quadros (opção no app), usar janelas maiores
  ou MobileNetV3 em lote.
* **Resolução de entrada:** reduzir de 224 para 96 px manteve F1 0,998 no teste, mas não acelerou na GPU
  (≈7 ms em todas as resoluções: o custo é de disparo, não de cálculo). Mantivemos 224 px pela margem para fissuras finas.
* O alerta de vídeo usa **média móvel + histerese**: liga após 3 quadros acima do limiar e desliga abaixo de
  0,8×limiar, para não piscar com um quadro ruim.

## 6. Integração ao CitizenTech (protótipo Zela)

O módulo foi incorporado ao **Zela**, o protótipo CitizenTech da equipe (Streamlit + API FastAPI). Ele segue a arquitetura
do Zela: o modelo roda **só no back-end** e o front-end apenas chama a API.

* **API** (`backend/app/vision/`, `routers/vision.py`): `POST /api/v1/vision/inspect` recebe uma foto e devolve a decisão,
  a confiança calibrada, as regiões marcadas, o Grad-CAM, a qualidade da foto e a latência. `POST /vision/sequence` e
  `POST /vision/monitor` (restritos à equipe da prefeitura) processam uma sequência de fotos ou um vídeo curto, com alertas
  por histerese e vídeo anotado. Sem PyTorch no servidor (ex.: Vercel), a API continua no ar e responde
  `503 vision_unavailable`. São 8 testes automatizados novos (7 na API, com um serviço falso no lugar do modelo, e 1 no front-end).
* **Cidadão:** no *Novo relato*, a foto é opcional. Na etapa *Revisar* aparece **"Possível fissura: inspeção recomendada"**
  com a confiança, a foto marcada e o mapa de atenção. O resumo da análise vai junto com o relato, e uma nova categoria,
  **🧱 Fissura em estrutura**, encaminha para a Defesa Civil. O cidadão continua com a palavra final (*human-in-the-loop*).
* **Prefeitura:** a fila mostra a foto anotada e a confiança. Entre relatos de mesma urgência e recorrência, os com
  fissura confirmada pela foto sobem: a foto **desempata**, não passa à frente da urgência. A aba **🎥 Monitoramento por
  câmera** simula o monitoramento urbano com vídeo ou sequência de fotos.
* **Medido dentro do Zela** (GTX 1650): foto de recorte em **91 ms** (inclui Grad-CAM e codificação da imagem); fachada
  3632×1362 (119 janelas) em **235 ms**; vídeo 1024×576 com 360 quadros a **10,2 FPS** de ponta a ponta (modelo: 57 ms por
  quadro), com 3 trechos para inspeção.

O projeto de Visão Computacional também traz um app de laboratório (`app/app.py`), usado para avaliar o modelo isolado.

## 7. Limitações e próximos passos

* A base tem recortes de **um único campus**, bem enquadrados. Fotos reais trazem perspectiva, escala variada,
  pintura, juntas de dilatação e pichação, que não foram testadas. É preciso coletar e rotular fotos da cidade
  antes de uso em produção.
* A localização é por **janela** (≈ recorte de 227 px), não por pixel. Para medir comprimento e abertura de
  fissura seria preciso segmentação (por exemplo, U-Net ou SAM, visto em aula).
* Incluir desfoque e baixa resolução no aumento de campo e reavaliar.
* A prioridade de triagem ordena vistorias. **Não é laudo**: a gravidade depende de inspeção técnica.
