# TCC Aura - Sistema Auditor com LLM e RAG

O **TCC Aura** é um sistema de auditoria automatizada que utiliza modelos de linguagem grandes (LLMs) via Ollama, juntamente com a técnica de Geração Aumentada por Recuperação (RAG) utilizando LangChain e ChromaDB. O foco do sistema é analisar documentos (corpus) fornecidos e avaliá-los contra controles e métricas de segurança baseadas no framework NIST.

## 🗂️ Estrutura do Projeto

```text
tcc-aura/
│
├── corpus/                 # Diretório para documentos a serem analisados (ex: .txt, .pdf)
├── data/                   # Diretório contendo as bases de controles (NIST) em CSV
│   ├── NIST_Requisitos.csv # Requisitos/Controles base
│   └── NIST_Definicoes.csv # Rubricas e critérios de avaliação para a auditoria
│
├── vector_store/           # Banco de dados vetorial gerado automaticamente (ChromaDB)
├── output/                 # Diretório de saída dos resultados (relatórios em JSONL/CSV)
│
├── ingestion.py            # Script responsável pelo processamento do corpus e criação da base vetorial
├── main.py                 # Ponto de entrada do sistema auditor principal
├── api.py                  # API para consumo do serviço (se aplicável)
├── database.py             # Configuração e integração com o banco de dados
└── testar_banco.py         # Script utilitário para validação da base
```

## 🛠️ Tecnologias e Dependências

- **Python:** Recomendada a versão `3.12.8`
- **LangChain:** Framework para orquestração da LLM e pipeline do RAG
- **Ollama:** Execução e inferência de LLMs localmente
- **ChromaDB:** Banco de dados vetorial
- **Pandas / Openpyxl:** Manipulação dos arquivos de entrada (CSV) e saída
- **PyPDF:** Processamento de documentos PDF no corpus

## 🚀 Como Configurar e Executar

### 1. Preparando o Ambiente

Recomenda-se utilizar um ambiente virtual (venv).

```bash
# Crie e ative o ambiente virtual
python -m venv .venv
source .venv/Scripts/activate  # No Windows (Git Bash / PowerShell)

# Instale os pacotes principais
pip install langchain-community langchain langchain-chroma pypdf langchain-ollama pandas openpyxl
```

_(Note que você pode precisar instalar dependências adicionais como clientes do MySQL caso utilize recursos específicos de banco relacional descritos no `database.py` e `mysqlconf.txt`)_

### 2. Alimentando os Dados

- Coloque os documentos que serão auditados na pasta `corpus/`.
- Certifique-se de que as tabelas de referência do NIST (`NIST_Requisitos.csv` e `NIST_Definicoes.csv`) estejam na pasta `data/`.

### 3. Ingestão e Vetorização

Execute o script de ingestão para ler os arquivos na pasta `corpus/` e popular o ChromaDB dentro da pasta `vector_store/`:

```bash
python ingestion.py
```

### 4. Executando a Auditoria

Após criar o banco vetorial, execute o arquivo principal para iniciar o fluxo de RAG e gerar o relatório final na pasta `output/`:

```bash
python main.py
```

## 📝 Notas Adicionais

- Certifique-se de ter o [Ollama](https://ollama.com/) devidamente instalado e o modelo utilizado rodando em sua máquina, caso utilize modelos de LLM servidos de modo local no seu ambiente.
- Verifique os arquivos `mysqlconf.txt` e script `mysql.sql` caso esteja utilizando integração para armazenamento persistente dos resultados/logs em banco de dados MySQL.
