# CitizenTech RPA + IA — Triagem Inteligente de Zeladoria e Serviços Públicos

**Disciplina:** AI for Robotic Process Automation (RPA + IA) — Global Solution 2026.2  
**Integrantes do Grupo:**
- Davi Gravina – RM 565619
- Henry Andrade Browne – RM 562089
- Luiz Felipe dos Santos Tragi – RM 565009

---

## 1. Descrição do Problema e Solução
Nas cidades brasileiras, prefeituras e órgãos públicos recebem diariamente milhares de solicitações cidadãs (buracos nas vias, falhas de iluminação pública, descarte irregular de lixo e vazamentos de esgoto) por canais não estruturados. A triagem manual gera gargalos operacionais, perda de protocolos, demora no atendimento de casos urgentes e falta de transparência para o morador.

Este protótipo automatiza o fluxo de ponta a ponta como um robô real:
1. **Ingere e higieniza** uma fila de solicitações cidadãs (`dados_cidadaos.csv`), removendo duplicidades e barrando registros corrompidos.
2. **Enriquece o endereço** automaticamente consultando a API pública ViaCEP (com *timeout*, *retry* e fallback).
3. **Classifica a categoria e a urgência** utilizando um modelo gratuito de Inteligência Artificial (NLP), encaminhando casos ambíguos para **revisão humana**.
4. **Persiste os protocolos** em um banco de dados relacional SQLite com garantia de idempotência.
5. **Notifica o cidadão** automaticamente com seu número de protocolo e gera um relatório auditável de execução e métricas.

---

## 2. Estrutura do Repositório

```text
RPA-Global-Solution/
├── .env.example            # Modelo de variáveis de ambiente (sem credenciais reais)
├── requirements.txt        # Lista de bibliotecas Python gratuitas utilizadas
├── dados_cidadaos.csv      # Dataset sintético rotulado para testes e validação
├── ia_triagem.py           # Módulo de IA (TF-IDF + Regressão Logística + regra de prioridade)
├── servicos_rpa.py         # Módulos RPA: API ViaCEP, Banco SQLite e Notificador de E-mail
├── main.py                 # Ponto de entrada único que orquestra todo o fluxo e gera logs
├── relatorio_validacao.md  # Relatório automático com métricas de acurácia e falhas tratadas
├── logs/
│   └── execucao_rpa.log    # Log detalhado de execução, alertas e exceções capturadas
└── emails_enviados/        # Comprovantes de protocolo gerados automaticamente (modo simulação)
```

---

## 3. Como Executar o Protótipo (Passo a Passo Completo)

### Pré-requisitos
- **Python 3.10 ou superior** instalado na máquina.
- Conexão com a internet para consulta à API pública gratuita **ViaCEP** (caso esteja offline, o robô possui fallback automático e continua funcionando sem quebrar).

### Etapa 1: Acessar a pasta do projeto
Abra o terminal (Prompt de Comando, PowerShell ou Terminal Linux/Mac) e navegue até a pasta raiz do repositório:
```bash
cd RPA-Global-Solution
```
Se o diretório local tiver sido renomeado, substitua o nome acima pelo nome da pasta em que o projeto foi clonado.

### Etapa 2: Criar e ativar o ambiente virtual (`venv`)
Isolar as dependências evita conflitos com outros projetos Python:

* **No Windows (PowerShell ou CMD):**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```
* **No Linux ou macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Etapa 3: Instalar as dependências
Com o ambiente virtual ativo, instale todas as bibliotecas listadas no `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Etapa 4: Configurar as variáveis de ambiente e chaves (`.env`)
Nenhuma credencial fica salva no código-fonte. Crie o arquivo `.env` local a partir do modelo `.env.example`:

* **No Windows:**
  ```bash
  copy .env.example .env
  ```
* **No Linux ou macOS:**
  ```bash
  cp .env.example .env
  ```

**Como configurar as chaves/variáveis dentro do `.env`:**
- `ARQUIVO_ENTRADA=dados_cidadaos.csv`: arquivo CSV com a fila de demandas sintéticas.
- `BANCO_SQLITE=citizentech_rpa.db`: nome do arquivo de banco de dados local que será criado.
- `LIMIAR_CONFIANCA_IA=0.55`: confiança mínima exigida da IA; abaixo disso, o pedido vai para **Revisão Humana**.
- `RESET_BANCO=true`: reinicia o banco para gerar uma demonstração limpa em cada execução; altere para `false` para preservar os protocolos existentes e testar idempotência entre execuções.
- `MODO_SIMULACAO_EMAIL=true`: **mantido como `true` por padrão** para permitir que qualquer avaliador rode o robô imediatamente sem precisar criar senhas SMTP. Os e-mails são salvos como arquivos `.txt` auditáveis na pasta `emails_enviados/`. *(Opcional: para disparo real via Gmail, altere para `false` e preencha `SMTP_USUARIO` e `SMTP_SENHA` com uma Senha de App).*

### Etapa 5: Executar o robô (Ponto de Entrada Único)
Rode o script principal para executar todo o pipeline do início ao fim sem intervenção manual:
```bash
python main.py
```

### Etapa 6: Verificar os resultados e evidências geradas
Após o término da execução (cerca de 2 a 5 segundos), confira os artefatos criados automaticamente na pasta:
1. **`relatorio_validacao.md`**: abra para ver o resumo operacional, a tabela de acertos/erros da IA e a lista de falhas tratadas.
2. **`logs/execucao_rpa.log`**: abra para inspecionar o passo a passo cronológico e os alertas de erro capturados.
3. **`citizentech_rpa.db`**: banco SQLite contendo a tabela `protocolos_cidadaos` populada.
4. **`emails_enviados/`**: pasta contendo um comprovante `.txt` para cada protocolo válido gerado.

---

## 4. Quadro de Temas da Disciplina Utilizados

| Tema da Disciplina | Onde é usado no fluxo | Qual problema específico resolve |
| :--- | :--- | :--- |
| **1. Tratamento de Arquivos e Dados (Pandas)** | Etapa inicial (`main.py`) na leitura do arquivo `dados_cidadaos.csv`. | Realiza a ingestão em lote, padroniza tipos de dados (preservando zeros à esquerda nos CEPs) e remove solicitações duplicadas na origem, garantindo **idempotência**. |
| **2. Integração com APIs REST (ViaCEP)** | Etapa de enriquecimento (`servicos_rpa.py -> ServicoViaCEP`). | Consulta a base pública do ViaCEP com *timeout* (4s), *retry* com espera e cache em memória para validar o CEP e preencher logradouro, bairro e cidade/UF automaticamente. |
| **3. Banco de Dados SQL (SQLite)** | Etapa de persistência (`servicos_rpa.py -> RepositorioProtocolosSQL`). | Armazena de forma estruturada e auditável os protocolos gerados e impede reprocessamento pelo `id_solicitacao`, inclusive entre execuções quando `RESET_BANCO=false`. |
| **4. E-mail Automatizado (SMTP / Simulação)** | Etapa final de resposta (`servicos_rpa.py -> NotificadorEmailRPA`). | Fecha o ciclo de atendimento enviando ao cidadão o número de protocolo, o setor responsável e a prioridade estimada do chamado sem intervenção manual. |

---

## 5. Evidências de Validação e Resiliência
Ao executar `python main.py` sobre a amostra sintética rotulada (`dados_cidadaos.csv`, 15 registros), o robô gera automaticamente o arquivo `relatorio_validacao.md` e o log `logs/execucao_rpa.log`, comprovando:
- **Acurácia da IA na amostra rotulada:** 50% de acerto nos 12 registros válidos avaliados (6 acertos e 6 erros). Solicitações ambíguas ou com confiança abaixo do limiar de `0.55`, incluindo `SOL-012`, são encaminhadas para **Revisão Humana**.
- **Idempotência:** O registro duplicado `SOL-001` é identificado e ignorado na entrada; o banco também usa `id_solicitacao` como chave primária e impede reprocessamento entre execuções quando o reinício do banco está desativado.
- **Tratamento de falhas previsíveis:**
  - *Arquivo ausente:* Simulado na inicialização (`arquivo_que_nao_existe.csv`), capturado via `FileNotFoundError` e registrado em log sem interromper o robô.
  - *CEP inexistente (`SOL-011`, CEP `99999999`):* Tratado pela API com fallback seguro (`CEP_NAO_ENCONTRADO`), mantendo o fluxo ativo.
  - *Dados corrompidos (`SOL-013` com CEP inválido e `SOL-014` com relato vazio):* Barrados na validação de integridade e registrados no log de alertas.
- **Limites conhecidos da solução:** O modelo TF-IDF depende de vocabulário textual próximo ao domínio de zeladoria urbana e não interpreta imagens anexadas pelos cidadãos; por isso, solicitações vagas ou fora do padrão são obrigatoriamente direcionadas para revisão humana.

---

## 6. Proteção de Dados (LGPD), Ética e ODS
Em conformidade com a Lei Geral de Proteção de Dados (LGPD), o protótipo aplica o princípio da minimização de dados, opera exclusivamente com registros sintéticos e dados públicos (ViaCEP), mantém credenciais isoladas em variáveis de ambiente (`.env`) e nunca envia informações pessoais reais a APIs externas. Sob a ótica ética, o sistema evita decisões totalmente opacas ao adotar um limiar de confiança auditável que encaminha casos ambíguos para revisão humana, prevenindo o descarte injusto de pedidos de cidadãos com menor letramento digital. Por fim, a solução contribui diretamente para os **ODS 11 (Cidades e Comunidades Sustentáveis)**, **ODS 9 (Indústria, Inovação e Infraestrutura)** e **ODS 16 (Paz, Justiça e Instituições Eficazes)** ao tornar os serviços urbanos mais ágeis, transparentes e acessíveis.
