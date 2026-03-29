"""
AURA - Automated Risk Assessment.
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

# --- Configurações Globais ---
DATA_DIR = "data"
OUTPUT_DIR = "output"
VECTOR_STORE_PATH = "vector_store"

INPUT_FILENAME = "NIST_Requisitos.csv"
RUBRIC_FILENAME = "NIST_Definicoes.csv"
REPORT_FILENAME = "Relatorio_Auditoria_Final.jsonl" # Saída em JSON Lines

LLM_MODEL = "llama3"

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
    """Configura a IA com instruções Céticas e saída em Array para recomendações."""
    
    llm = ChatOllama(model=LLM_MODEL, temperature=0, format="json")
    
    audit_template = """
    Você é um Auditor Extremamente Cético do NIST CSF 2.0.
    
    SUA MISSÃO:
    Avaliar se existem provas concretas para o controle abaixo. 
    Se a prova for vaga, genérica ou não citar explicitamente o tema do controle, você DEVE rejeitar.
    
    GABARITO DE MATURIDADE:
    {rules}
    
    CONTROLE ALVO:
    ID: {subcat}
    Descrição: {desc}
    
    CONTEXTO RECUPERADO DOS ARQUIVOS:
    {context}
    
    PROTOCOLO DE RECUSA:
    1. Se o Contexto fala de "Segurança" no geral, mas o Controle pede algo específico, RESPOSTA: "Não Avaliado".
    2. Se o Contexto não tem relação direta com o ID {subcat}, RESPOSTA: "Não Avaliado".
    3. NÃO TENTE AJUDAR. Seja frio e literal.
    
    FORMATO DE SAÍDA OBRIGATÓRIO (JSON):
    {{
        "nivel": "String ('Nível X...' ou 'Não Avaliado')",
        "pontuacao": Inteiro de 1-4 (0 se Não Avaliado),
        "cenario": "Explique a situação encontrada com base no contexto, seja específico e cite o que foi encontrado ou não encontrado.",
        "evidencia": "Citação exata + [FONTE] (Deixe vazio se Não Avaliado)",
        "gaps": "Liste o que falta, ou esteja incompleto em relação a essa subcategoria, por exemplo, falta de elementos necessários na documentação relacionada a subcategoria, ou realmente falta na prática, lembre-se de não se limitar apenas a esses exemplos. (Se não avaliado, diga 'Sem dados para análise')",
        "recomendacoes": ["Ação prática 1", "Ação prática 2"], visando detalhar todas as ações necessárias para avançar ao próximo nível de maturidade.
    }}
    NOTA IMPORTANTE: O campo 'recomendacoes' DEVE SER UMA LISTA DE STRINGS. Mesmo se houver apenas uma, coloque dentro de colchetes.
    """
    
    prompt = PromptTemplate(
        template=audit_template,
        input_variables=["rules", "subcat", "desc", "context"]
    )
    
    return prompt | llm | JsonOutputParser()

def execute_audit_process():
    print(f"[*] Iniciando Auditoria Cética NIST v2.0 (Exportação JSONL Rastreável)...")
    
    if not os.path.exists(VECTOR_STORE_PATH):
        sys.exit("[ERROR] Banco de dados não encontrado. Rode 'ingestion.py' primeiro.")

    # 1. Conexão com o Banco Vetorial
    embedding_fn = OllamaEmbeddings(model=LLM_MODEL)
    vector_db = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embedding_fn)
    
    rubric_path = os.path.join(DATA_DIR, RUBRIC_FILENAME)
    rubrics_context = load_assessment_rubrics(rubric_path)
    
    audit_chain = get_skeptical_auditor_chain(rubrics_context)

    # 2. Carregar Perguntas do NIST
    input_path = os.path.join(DATA_DIR, INPUT_FILENAME)
    try:
        df_controls = pd.read_csv(input_path, header=1, sep=';', encoding='utf-8')
        target_controls = df_controls[df_controls['Subcategoria'].notna()].copy()
    except Exception as e:
        sys.exit(f"[ERROR] Erro ao ler planilha base: {e}")

    total = len(target_controls)
    print(f"[*] Total de controles a verificar: {total}")

    # --- VARIÁVEIS DE RASTREABILIDADE DO LOTE ---
    # Gera um ID único para ESTA rodada de auditoria inteira
    id_lote_auditoria = str(uuid.uuid4())
    data_hora_atual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    jsonl_records = []
    count = 0
    
    # 3. Loop Principal de Auditoria
    for idx, row in target_controls.iterrows():
        count += 1
        funcao = row.get('Função', 'Desconhecida')
        categoria = row.get('Categoria', 'Desconhecida')
        cid = row['Subcategoria']
        cdesc = row['Descrição da subcategoria']
        
        print(f"    [{count}/{total}] Auditando {cid}...", end="\r")
        
        # Recuperação de Documentos (RAG)
        query = f"{cid} {cdesc} evidência prática política implementação"
        docs = vector_db.similarity_search(query, k=5)
        
        context_parts = []
        for d in docs:
            source_name = os.path.basename(d.metadata.get('source', 'Arquivo Desconhecido'))
            context_parts.append(f"[[FONTE: {source_name}]]\nCONTEÚDO: {d.page_content}")
            
        context_str = "\n---\n".join(context_parts)
        
        try:
            # Inferência da IA
            result = audit_chain.invoke({
                "rules": rubrics_context,
                "subcat": cid,
                "desc": cdesc,
                "context": context_str
            })
            
            # --- LÓGICA DE SEGURANÇA E TRATAMENTO DE DADOS ---
            nivel = str(result.get('nivel', ''))
            
            if "Não Avaliado" in nivel or "N/A" in nivel:
                pontuacao = 0
                evidencia = ""
                recomendacoes_lista = ["Realizar entrevista ou buscar documentos sobre este tópico."]
            else:
                pontuacao = result.get('pontuacao', 0)
                evidencia = str(result.get('evidencia', ''))
                
                # Garante que as recomendações sejam sempre uma lista (Array)
                recs_brutas = result.get('recomendacoes', [])
                if isinstance(recs_brutas, str):
                    recomendacoes_lista = [recs_brutas]
                elif isinstance(recs_brutas, list):
                    recomendacoes_lista = recs_brutas
                else:
                    recomendacoes_lista = ["Revisar Gaps apontados"]

            # --- CONSTRUÇÃO DO REGISTRO JSON COM RASTREABILIDADE ---
            record = {
                "registro_id": str(uuid.uuid4()),            # ID Único Desta Avaliação Específica
                "auditoria_id": id_lote_auditoria,           # ID do Lote (Agrupa toda a rodada)
                "data_avaliacao": data_hora_atual,           # Timestamp da execução
                "funcao": funcao,
                "categoria": categoria,
                "subcategoria_id": cid,
                "descricao": cdesc,
                "nivel_maturidade": nivel,
                "pontuacao": pontuacao,
                "cenario_atual": str(result.get('cenario', '')),
                "evidencia": evidencia,
                "gaps": str(result.get('gaps', '')),
                "recomendacoes": recomendacoes_lista         # Campo formatado como Lista
            }
            
            jsonl_records.append(record)
            
        except Exception as e:
            print(f"\n[WARN] Erro no controle {cid}: {e}")

    # --- 4. EXPORTAÇÃO JSONL ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    final_path = os.path.join(OUTPUT_DIR, REPORT_FILENAME)
    
    # Salva o arquivo escrevendo uma linha por vez (padrão JSONL)
    with open(final_path, 'w', encoding='utf-8') as f:
        for record in jsonl_records:
            # ensure_ascii=False garante que acentos fiquem legíveis e não codificados
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"\n\n[SUCCESS] Relatório JSONL salvo em: {final_path}")
    print(f"[INFO] ID do Lote de Auditoria gerado: {id_lote_auditoria}")

if __name__ == "__main__":
    execute_audit_process()