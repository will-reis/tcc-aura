import mysql.connector
import json
import bcrypt

# --- CONFIGURAÇÕES DE SEGURANÇA ---
# Em um ambiente corporativo real, essas variáveis viriam de um arquivo .env oculto.
DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = "admin"
DB_NAME = "nist_audit_db"

def verify_user(username, password):
    """
    Verifica as credenciais do usuário com o banco de dados.
    """
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, username, password_hash, role FROM usuarios WHERE username = %s",
            (username,)
        )
        user = cursor.fetchone()
        
        if user:
            # Verifica a senha usando bcrypt
            if bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
                # Remove o hash da senha antes de retornar os dados
                del user['password_hash']
                return user
                
        return None
    except Exception as e:
        print(f"[ERROR] Falha ao verificar usuário: {e}")
        return None
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def _normalize_recomendacoes(raw_value):
    """Normaliza o campo recomendacoes para lista de strings."""
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return raw_value
    if isinstance(raw_value, (bytes, bytearray)):
        raw_value = raw_value.decode("utf-8", errors="ignore")
    if isinstance(raw_value, str):
        value = raw_value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else [str(parsed)]
        except json.JSONDecodeError:
            return [value]
    return [str(raw_value)]

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


def get_all_assessments(include_records=True):
    """Busca todas as auditorias salvas, com opção de incluir os registros detalhados."""
    conn = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
    )

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT id, auditoria_uuid, usuario_id, data_avaliacao
            FROM assessments
            ORDER BY data_avaliacao DESC, id DESC
            """
        )
        assessments = cursor.fetchall()

        if not assessments:
            return []

        if not include_records:
            return [
                {
                    "assessment_id": item["id"],
                    "auditoria_id": item["auditoria_uuid"],
                    "usuario_id": item["usuario_id"],
                    "data_avaliacao": item["data_avaliacao"].isoformat(sep=" ") if item["data_avaliacao"] else None,
                }
                for item in assessments
            ]

        assessment_ids = [item["id"] for item in assessments]
        placeholders = ", ".join(["%s"] * len(assessment_ids))
        cursor.execute(
            f"""
            SELECT
                assessment_id,
                registro_uuid,
                funcao,
                categoria,
                subcategoria_id,
                descricao,
                nivel_maturidade,
                pontuacao,
                cenario_atual,
                evidencia,
                gaps,
                recomendacoes
            FROM resultados_nist
            WHERE assessment_id IN ({placeholders})
            ORDER BY id ASC
            """,
            tuple(assessment_ids),
        )
        raw_records = cursor.fetchall()

        records_by_assessment = {}
        for rec in raw_records:
            aid = rec["assessment_id"]
            records_by_assessment.setdefault(aid, []).append(
                {
                    "registro_id": rec["registro_uuid"],
                    "funcao": rec["funcao"],
                    "categoria": rec["categoria"],
                    "subcategoria_id": rec["subcategoria_id"],
                    "descricao": rec["descricao"],
                    "nivel_maturidade": rec["nivel_maturidade"],
                    "pontuacao": rec["pontuacao"],
                    "cenario_atual": rec["cenario_atual"],
                    "evidencia": rec["evidencia"],
                    "gaps": rec["gaps"],
                    "recomendacoes": _normalize_recomendacoes(rec["recomendacoes"]),
                }
            )

        response = []
        for item in assessments:
            aid = item["id"]
            registros = records_by_assessment.get(aid, [])
            response.append(
                {
                    "assessment_id": aid,
                    "auditoria_id": item["auditoria_uuid"],
                    "usuario_id": item["usuario_id"],
                    "data_avaliacao": item["data_avaliacao"].isoformat(sep=" ") if item["data_avaliacao"] else None,
                    "total_registros": len(registros),
                    "registros": registros,
                }
            )

        return response
    finally:
        conn.close()