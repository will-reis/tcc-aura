"""
AURA - Automated Risk Assessment.
Versão Modular (IA separada do Banco de Dados).
"""

import os
import sys
import pandas as pd
import json
import uuid
from datetime import datetime
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# 🌟 IMPORTAÇÃO DO NOSSO NOVO MÓDULO DE BANCO DE DADOS 🌟
from database import save_to_mysql

# --- Configurações Globais ---
DATA_DIR = "data"
OUTPUT_DIR = "output"
VECTOR_STORE_PATH = "vector_store"

INPUT_FILENAME = "NIST_Requisitos.csv"
RUBRIC_FILENAME = "NIST_Definicoes.csv"
REPORT_FILENAME = "Relatorio_Auditoria_Final.jsonl"

# Lembre-se de mudar aqui para o modelo Qwen que você baixou
LLM_MODEL = "qwen3:8b"
EMBEDDING_MODEL = "nomic-embed-text"

def load_assessment_rubrics(filepath: str) -> str:
    """Carrega as definições de maturidade (Gabarito)."""
    if not os.path.exists(filepath):
        sys.exit(f"[FATAL] Arquivo de definições não encontrado: {filepath}")

    try:
        df = pd.read_csv(filepath, header=2, sep=';', encoding='utf-8')
        rules_buffer = []
        for _, row in df.iterrows():
            if pd.notna(row.get('Nível de maturidade')):
                level = row['Nível de maturidade']
                criteria = row.get('Governança de Riscos de Cibersegurança', 'N/A')
                rules_buffer.append(f"- {level}: {criteria}")
        return "\n".join(rules_buffer)
    except Exception as e:
        sys.exit(f"[FATAL] Erro ao ler definições: {e}")

def get_skeptical_auditor_chain(rules_context):
    """Configura a IA com instruções Céticas e saída em JSON estruturado."""
    llm = ChatOllama(model=LLM_MODEL, temperature=0, format="json")
    
    audit_template = """
    Você é um Auditor Sênior e Especialista no framework NIST CSF 2.0.
    Sua análise é baseada estritamente no CONTEXTO fornecido, sendo rigoroso.

    ATENÇÃO - TRATAMENTO DE TRANSCRIÇÕES:
    As principais fontes podem ser transcrições de entrevistas geradas por IA. Tolere erros de digitação, falhas de concordância ou palavras sem sentido (alucinações de áudio), focando sempre na intenção e no contexto técnico da conversa para extrair as evidências.

    GABARITO DE MATURIDADE:
    {rules}
    
    CONTROLE ALVO:
    ID: {subcat}
    Descrição: {desc}
    
    CONTEXTO RECUPERADO:
    {context}
    
    REGRAS DE PONTUAÇÃO (Siga estritamente esta ordem lógica):
    * NOTA 4 (Nível 4): Extremamente perfeito. A entrevista confirma a aplicação prática do controle, há documentação completa detalhando o processo, e NÃO há nenhum gap identificado (teórico ou prático), ou seja, caso exista um gap a nota já não poderá ser 4, mas deve ser inferior.
    * NOTA 3 (Nível 3): A entrevista/contexto confirma que o assunto da subcategoria ESTÁ SENDO APLICADO na prática com base em um processo ou procedimento definido e documentado,mesmo que a documentação tenha pequenos gaps.
    * NOTA 2 (Nível 2): O processo NÃO foi citado como aplicado na prática, PORÉM existe documentação ou procedimento formal definindo como aquele tópico deve ser feito.
    * NOTA 1 (Nível 1): Pior cenário. Não há citação de aplicação prática e não há nenhuma definição, política ou documentação sobre o tópico em lugar algum.
    
    DIRETRIZES DE SAÍDA:
    - CENÁRIO ATUAL: Descreva a situação atual da organização.
    - EVIDÊNCIA: Se houver, descreva o trecho que baseou sua nota e liste TODAS as fontes (ex: [FONTE: doc1.pdf, transcricao.txt]).
    - GAPS: Liste todas as falhas na atuação prática ou falta de detalhamento em documentos. (Se a nota for 4, diga "Nenhum Gap").
    - RECOMENDAÇÕES: Forneça um plano de ação para atingir o PRÓXIMO nível de maturidade (Ex: Se tirou nota 2, o que falta para tirar nota 3?). DEVE SER UM ARRAY (LISTA) DE STRINGS.
    
    FORMATO OBRIGATÓRIO DE SAÍDA (JSON PURO):
    Você é uma API. Retorne APENAS um objeto JSON válido. 
    NÃO inclua texto antes ou depois. 
    NÃO use blocos de código markdown (como ```json).
    A resposta deve começar com {{ e terminar com }}.
    
    {{
        "nivel": "Nível X... (ou Não Avaliado)",
        "pontuacao": 0,
        "cenario": "...",
        "evidencia": "...",
        "gaps": "...",
        "recomendacoes": ["Ação 1", "Ação 2"]
    }}
    """
    
    prompt = PromptTemplate(
        template=audit_template,
        input_variables=["rules", "subcat", "desc", "context"]
    )
    
    return prompt | llm | JsonOutputParser()

def execute_audit_process():
    print(f"[*] Iniciando Auditoria NIST CSF v2.0 com modelo de IA {LLM_MODEL}...")
    
    if not os.path.exists(VECTOR_STORE_PATH):
        sys.exit("[ERROR] Banco de dados vetorial não encontrado.")

    embedding_fn = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_db = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embedding_fn)
    
    rubric_path = os.path.join(DATA_DIR, RUBRIC_FILENAME)
    rubrics_context = load_assessment_rubrics(rubric_path)
    
    audit_chain = get_skeptical_auditor_chain(rubrics_context)

    input_path = os.path.join(DATA_DIR, INPUT_FILENAME)
    try:
        df_controls = pd.read_csv(input_path, header=1, sep=';', encoding='utf-8')
        target_controls = df_controls[df_controls['Subcategoria'].notna()].copy()
    except Exception as e:
        sys.exit(f"[ERROR] Erro ao ler planilha base: {e}")

    total = len(target_controls)
    id_lote_auditoria = str(uuid.uuid4())
    data_hora_atual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    jsonl_records = []
    count = 0
    
    for idx, row in target_controls.iterrows():
        count += 1
        cid = row['Subcategoria']
        cdesc = row['Descrição da subcategoria']
        print(f"    [{count}/{total}] Auditando {cid}...", end="\r")
        
        docs = vector_db.similarity_search(f"{cid} {cdesc}", k=5)
        context_str = "\n---\n".join([f"[[FONTE: {os.path.basename(d.metadata.get('source', 'N/A'))}]]\n{d.page_content}" for d in docs])
        
        try:
            result = audit_chain.invoke({"rules": rubrics_context, "subcat": cid, "desc": cdesc, "context": context_str})
            
            # Formatação do registro
            record = {
                "registro_id": str(uuid.uuid4()),
                "auditoria_id": id_lote_auditoria,
                "data_avaliacao": data_hora_atual,
                "funcao": row.get('Função', 'N/A'),
                "categoria": row.get('Categoria', 'N/A'),
                "subcategoria_id": cid,
                "descricao": cdesc,
                "nivel_maturidade": result.get('nivel', 'Não Avaliado'),
                "pontuacao": result.get('pontuacao', 0),
                "cenario_atual": result.get('cenario', ''),
                "evidencia": result.get('evidencia', ''),
                "gaps": str(result.get('gaps', '')),
                "recomendacoes": result.get('recomendacoes', [])
            }
            jsonl_records.append(record)
        except Exception as e:
            print(f"\n[WARN] Erro em {cid}: {e}")

    # Salvar JSONL
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    final_path = os.path.join(OUTPUT_DIR, REPORT_FILENAME)
    with open(final_path, 'w', encoding='utf-8') as f:
        for record in jsonl_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    # SALVAR NO MYSQL
    if jsonl_records:
        save_to_mysql(jsonl_records)
    
    print(f"\n\n[SUCCESS] Auditoria Finalizada. Resultados em {final_path} e no Banco de Dados.")

if __name__ == "__main__":
    execute_audit_process()