# CitizenTech GenAI — Assistente Conversacional de Zeladoria Cidadã (RAG)

**Disciplina:** Generative AI & Advanced Nets — Global Solution 2026.2 (FIAP)  
**Turma:** 2TIAPF  
**Integrantes do Grupo:**
- **Davi Gravina** — RM 565619
- **Henry Andrade Browne** — RM 562089
- **Luiz Felipe dos Santos Tragi** — RM 565009

---

## 1. Visão Geral do Projeto
O **Assistente Conversacional CitizenTech** integra-se à solução de triagem inteligente de serviços públicos do grupo para orientar cidadãos sobre zeladoria urbana (**Tapa-Buraco/Infraestrutura**, **Iluminação Pública**, **Limpeza Urbana/Coleta**, **Vigilância Sanitária/Dengue** e **Acompanhamento de Protocolos**).

Diferente de um chatbot genérico, o sistema implementa uma arquitetura **RAG (Retrieval-Augmented Generation)** auditável com:
- **Base de Conhecimento Rastreável (`corpus/`):** Documentos do projeto registram fonte declarada, URL, categoria, data de consulta, órgão e prazos. O código não verifica se essas referências ou informações continuam válidas; valide-as antes de usar o conteúdo para orientar cidadãos.
- **Chunking Estrutural:** Segmenta por seções Markdown, com limite de 650 caracteres e overlap de 120 caracteres quando a seção excede esse limite. Metadados de fonte, URL, data e categoria acompanham cada trecho.
- **Retriever Híbrido (BM25 + Vetorial TF-IDF):** Combina busca lexical e similaridade de cosseno (pesos `0.45` e `0.55`), com stemming em português, `top-k=3` e limiar de confiança `0.22`.
- **Generator com System Prompt:** Respostas sustentadas pelo corpus devem citar fonte, URL e data; quando não há evidência suficiente, o sistema se abstém.
- **Guardrails de Segurança e LGPD (`seguranca.py`):** Detecta padrões de *Prompt Injection* direto e indireto e mascara CPF, e-mail e telefone. São mitigações baseadas em regras, não uma garantia de proteção contra todo ataque. Credenciais opcionais ficam no `.env`, ignorado pelo Git.

---

## 2. Estrutura do Repositório

```text
Global-Solution_Genai/
├── .gitignore                          # Ignora configurações locais do VS Code
├── README.md                           # Este guia de instalação e uso
└── citizentech-genai/
    ├── .gitignore                      # Bloqueia .env, ambientes virtuais e caches
    ├── .env.example                    # Modelo de configuração sem credenciais reais
    ├── requirements.txt                # Dependências Python
    ├── corpus/                         # Base de conhecimento rastreável
    ├── seguranca.py                    # Guardrails: LGPD, injection e citação
    ├── rag_pipeline.py                 # Chunking, Retriever Híbrido e Generator RAG
    ├── app_chat.py                     # Interface conversacional interativa (CLI)
    ├── avaliar_rag.py                  # Bateria automatizada de 12 testes e métricas
    ├── DOCUMENTACAO_CORPUS_E_ARQUITETURA.md
    └── RELATORIO_AVALIACAO.md          # Gerado por avaliar_rag.py
```

---

## 3. Como Instalar e Executar (Passo a Passo)

### Pré-requisitos
- **Python 3.10 ou superior** instalado.
- Não é necessária chave de API: `LLM_PROVIDER=fallback` é o padrão e usa um gerador determinístico local, sem chamada a serviço externo.
- Os testes e exemplos abaixo devem ser executados a partir da raiz do repositório, seguindo o comando `cd citizentech-genai`.

### Etapa 1: Acessar a pasta e criar o ambiente virtual
```bash
cd citizentech-genai
python -m venv venv

# Ativar no Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Ativar no Windows Command Prompt (CMD):
venv\Scripts\activate.bat

# Ativar no Linux ou macOS:
source venv/bin/activate
```

### Etapa 2: Instalar as dependências
```bash
pip install -r requirements.txt
```

### Etapa 3: Configurar variáveis de ambiente
Copie o modelo `.env.example` para `.env`:
```bash
# Windows:
copy .env.example .env

# Linux/macOS:
cp .env.example .env
```
O modo fallback funciona sem chave. Opcionalmente:
- **Groq:** configure `LLM_PROVIDER=groq` e uma chave válida em `GROQ_API_KEY`; requer acesso à internet e está sujeito à disponibilidade e aos limites do serviço.
- **Ollama:** configure `LLM_PROVIDER=ollama`, instale/inicie o Ollama localmente e disponibilize o modelo configurado em `OLLAMA_MODEL`.

### Etapa 4: Executar a Bateria de Testes e Gerar o Relatório de Avaliação
```bash
python avaliar_rag.py
```
Este comando executa a comparação entre duas estratégias de Chunking, roda os **12 casos de teste** (respondíveis, sem resposta, ambíguos, multi-turno, fora de escopo e 2 testes adversariais) e gera/atualiza o arquivo **`RELATORIO_AVALIACAO.md`**.
Os resultados são evidências desta bateria fixa de testes, não uma certificação geral da precisão ou segurança do sistema.

### Etapa 5: Testar o Assistente Conversacional Interativamente
```bash
python app_chat.py
```
Digite suas perguntas no terminal para visualizar os trechos recuperados, scores híbridos, alertas de segurança e respostas. Use `/limpar` para reiniciar o histórico ou `/sair` para encerrar.

---

## 4. Documentação Complementar
- **Justificativas Técnicas, LGPD e ODS:** Consulte [`DOCUMENTACAO_CORPUS_E_ARQUITETURA.md`](citizentech-genai/DOCUMENTACAO_CORPUS_E_ARQUITETURA.md).
- **Evidências de Testes e Métricas:** Geradas em `citizentech-genai/RELATORIO_AVALIACAO.md` ao executar a avaliação.
