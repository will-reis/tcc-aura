import json
import os
from database import save_to_mysql

# Caminho onde o seu relatório JSONL já está salvo
ARQUIVO_JSONL = os.path.join("output", "Relatorio_Auditoria_Final.jsonl")

def popular_banco_direto_do_arquivo():
    print(f"[*] Procurando arquivo: {ARQUIVO_JSONL}")
    
    if not os.path.exists(ARQUIVO_JSONL):
        print("[ERRO] Arquivo JSONL não encontrado! Certifique-se de que ele está na pasta 'output'.")
        return

    # Lista que vai guardar os dados lidos do arquivo
    jsonl_records = []
    
    # 1. Abre o arquivo e lê linha por linha
    with open(ARQUIVO_JSONL, 'r', encoding='utf-8') as f:
        for linha in f:
            if linha.strip(): # Ignora linhas vazias
                # Converte a linha de texto JSON para um Dicionário Python
                jsonl_records.append(json.loads(linha))
                
    print(f"[*] {len(jsonl_records)} registros carregados do arquivo com sucesso.")
    
    # 2. Envia para o MySQL usando a função do database.py
    if jsonl_records:
        sucesso = save_to_mysql(jsonl_records) # O user_id=1 já é o padrão da função
        if sucesso:
            print("\n[MÁGICA FEITA] ✨ Todos os dados foram para o MySQL sem precisar rodar a IA de novo!")

if __name__ == "__main__":
    popular_banco_direto_do_arquivo()