"""
NIST CSF 2.0 Compliance Audit Engine - Versão Final (Rastreabilidade + Recomendações).
"""

import os
import sys
import pandas as pd
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# --- Configurações ---
DATA_DIR = "data"
OUTPUT_DIR = "output"
VECTOR_STORE_PATH = "vector_store"

INPUT_FILENAME = "NIST_Requisitos.csv"
RUBRIC_FILENAME = "NIST_Definicoes.csv"
REPORT_FILENAME = "Relatorio_Auditoria_Final.csv"

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

def get_strict_auditor_chain(rules_context):
    """Configura o cérebro da IA com instruções para Fonte e Recomendações."""
    
    llm = ChatOllama(model=LLM_MODEL, temperature=0, format="json")
    
    audit_template = """
    Você é um Auditor Rígido do NIST CSF 2.0.
    
    SUA TAREFA:
    Preencher a planilha de avaliação com base ESTRITAMENTE nas evidências fornecidas.
    
    REGRAS DE MATURIDADE (GABARITO):
    {rules}
    
    CONTROLE A SER AUDITADO:
    ID: {subcat}
    Descrição: {desc}
    
    EVIDÊNCIAS DOS ARQUIVOS (CONTEXTO COM FONTES):
    {context}
    
    DIRETRIZES OBRIGATÓRIAS:
    1. IDIOMA: Responda SEMPRE em PORTUGUÊS do Brasil.
    2. RASTREABILIDADE: Na evidência, você DEVE citar o nome do arquivo de onde tirou a informação (ex: "[FONTE: entrevista.txt] O usuário disse...").
    3. RECOMENDAÇÃO: Se houver Gaps, sugira uma ação prática para resolvê-lo (ex: "Elaborar política formal", "Implementar MFA").
    4. HONESTIDADE: Se não houver menção no contexto, responda "Não Avaliado".
    5. Se o Contexto não tem relação direta com o ID {subcat}, RESPOSTA: "Não Avaliado".
    6. NÃO TENTE AJUDAR. Seja frio e literal.

    FORMATO DE SAÍDA (JSON):
    {{
        "nivel": "String (ex: Nível 1: Parcial)",
        "pontuacao": Inteiro (ex: 1),
        "cenario": "Resumo da situação encontrada...",
        "evidencia": "Citação do texto + [FONTE: Nome do Arquivo]",
        "gaps": "O que falta para o próximo nível...",
        "recomendacao": "Ação corretiva sugerida..."
    }}
    """
    
    prompt = PromptTemplate(
        template=audit_template,
        input_variables=["rules", "subcat", "desc", "context"]
    )
    
    return prompt | llm | JsonOutputParser()

def execute_audit_process():
    print(f"[*] Iniciando Auditoria NIST CSF v2.0...")
    
    if not os.path.exists(VECTOR_STORE_PATH):
        sys.exit("[ERROR] Banco de dados não encontrado. Rode 'ingestion.py' primeiro.")

    embedding_fn = OllamaEmbeddings(model=LLM_MODEL)
    vector_db = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embedding_fn)
    
    rubric_path = os.path.join(DATA_DIR, RUBRIC_FILENAME)
    rubrics_context = load_assessment_rubrics(rubric_path)
    
    audit_chain = get_strict_auditor_chain(rubrics_context)

    # 2. Carregar Planilha
    input_path = os.path.join(DATA_DIR, INPUT_FILENAME)
    try:
        df_controls = pd.read_csv(input_path, header=1, sep=';', encoding='utf-8')
        
        # --- PREPARAÇÃO DAS COLUNAS (Fix Float64) ---
        colunas_alvo = [
            'Nivel de Maturidade Atual', 
            'Cenário Atual', 
            'Evidência/Fonte', 
            'Gaps', 
            'Recomendações'
        ]
        
        # Converte para Object (Texto) para evitar erros
        for col in colunas_alvo:
            if col in df_controls.columns:
                df_controls[col] = df_controls[col].astype('object')
            else:
                # Se a coluna não existir, cria ela vazia
                df_controls[col] = None

        # Remove colunas lixo (Unnamed)
        df_controls = df_controls.loc[:, ~df_controls.columns.str.contains('^Unnamed')]
        
        target_controls = df_controls[df_controls['Subcategoria'].notna()].copy()

    except Exception as e:
        sys.exit(f"[ERROR] Erro ao ler planilha de requisitos: {e}")

    total = len(target_controls)
    print(f"[*] Total de controles a verificar: {total}")

    count = 0
    for idx, row in target_controls.iterrows():
        count += 1
        cid = row['Subcategoria']
        cdesc = row['Descrição da subcategoria']
        
        print(f"    [{count}/{total}] Auditando {cid}...", end="\r")
        
        # --- BUSCA INTELIGENTE COM FONTE ---
        query = f"{cid} {cdesc} evidência prática política implementação"
        docs = vector_db.similarity_search(query, k=4)
        
        # Monta o contexto inserindo o nome do arquivo explicitamente
        context_parts = []
        for d in docs:
            # Pega o nome do arquivo nos metadados (ou 'Desconhecido')
            source_name = os.path.basename(d.metadata.get('source', 'Arquivo Desconhecido'))
            context_parts.append(f"[[FONTE: {source_name}]]\nCONTEÚDO: {d.page_content}")
            
        context_str = "\n---\n".join(context_parts)
        
        try:
            result = audit_chain.invoke({
                "rules": rubrics_context,
                "subcat": cid,
                "desc": cdesc,
                "context": context_str
            })
            
            # --- PREENCHIMENTO ---
            df_controls.at[idx, 'Nivel de Maturidade Atual'] = str(result.get('nivel'))
            df_controls.at[idx, 'Pontuação Atual'] = result.get('pontuacao')
            df_controls.at[idx, 'Cenário Atual'] = str(result.get('cenario'))
            df_controls.at[idx, 'Evidência/Fonte'] = str(result.get('evidencia'))
            df_controls.at[idx, 'Gaps'] = str(result.get('gaps'))
            # Agora preenche a recomendação real
            df_controls.at[idx, 'Recomendações'] = str(result.get('recomendacao'))
            
        except Exception as e:
            print(f"\n[WARN] Erro no controle {cid}: {e}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    final_path = os.path.join(OUTPUT_DIR, REPORT_FILENAME)
    df_controls.to_csv(final_path, index=False, sep=';', encoding='utf-8-sig')
    
    print(f"\n\n[SUCCESS] Relatório salvo em: {final_path}")

if __name__ == "__main__":
    execute_audit_process()