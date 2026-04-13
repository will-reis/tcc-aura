import mysql.connector
import json

# --- CONFIGURAÇÕES DE SEGURANÇA ---
# Em um ambiente corporativo real, essas variáveis viriam de um arquivo .env oculto.
DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = "admin"
DB_NAME = "nist_audit_db"

def save_to_mysql(json_records, user_id=1):
    """
    Recebe a lista de dicionários JSONL e salva nas tabelas do MySQL.
    Retorna True se for bem-sucedido, False se falhar.
    """
    try:
        print("[*] Conectando ao MySQL para salvar resultados...")
        
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME
        )
        cursor = conn.cursor()

        # 1. Registrar o Assessment (Cabeçalho do Lote)
        lote_id = json_records[0]['auditoria_id']
        data_av = json_records[0]['data_avaliacao']
        
        sql_assess = "INSERT INTO assessments (auditoria_uuid, usuario_id, data_avaliacao) VALUES (%s, %s, %s)"
        cursor.execute(sql_assess, (lote_id, user_id, data_av))
        assessment_db_id = cursor.lastrowid

        # 2. Registrar cada linha de resultado (Detalhes)
        sql_res = """INSERT INTO resultados_nist 
                     (assessment_id, registro_uuid, funcao, categoria, subcategoria_id, descricao, 
                      nivel_maturidade, pontuacao, cenario_atual, evidencia, gaps, recomendacoes) 
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""

        for rec in json_records:
            cursor.execute(sql_res, (
                assessment_db_id,
                rec['registro_id'],
                rec['funcao'],
                rec['categoria'],
                rec['subcategoria_id'],
                rec['descricao'],
                rec['nivel_maturidade'],
                rec['pontuacao'],
                rec['cenario_atual'],
                rec['evidencia'],
                rec['gaps'],
                json.dumps(rec['recomendacoes'], ensure_ascii=False) # Converte a lista para string JSON para o MySQL
            ))

        conn.commit()
        print(f"[SUCCESS] {len(json_records)} registros salvos com sucesso no Banco de Dados.")
        return True
        
    except Exception as e:
        print(f"[ERROR] Falha crítica ao salvar no banco MySQL: {e}")
        return False
        
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()