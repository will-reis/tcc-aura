from pathlib import Path
from typing import List
import json
import shutil
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="AuraAPI")

class LoginRequest(BaseModel):
    username: str
    password: str

CORPUS_PATH = Path("corpus")
REPORT_PATH = Path("output") / "Relatorio_Auditoria_Final.jsonl"
AUDIT_HISTORY_DIR = Path("output") / "auditorias"
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def load_report_records(report_path: Path) -> list[dict]:
    """Carrega os registros de auditoria no formato JSONL."""
    records = []
    with report_path.open("r", encoding="utf-8") as report_file:
        for line in report_file:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def persist_audit_history(audit_payload: dict) -> Path:
    """Salva uma cópia do resultado para permitir consulta histórica via API."""
    AUDIT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    audit_id = audit_payload.get("auditoria_id") or str(uuid.uuid4())
    target_path = AUDIT_HISTORY_DIR / f"{audit_id}.json"
    with target_path.open("w", encoding="utf-8") as history_file:
        json.dump(audit_payload, history_file, ensure_ascii=False, indent=2)
    return target_path


def load_audit_history_from_files() -> list[dict]:
    """Carrega todas as auditorias salvas em disco, da mais recente para a mais antiga."""
    if not AUDIT_HISTORY_DIR.exists():
        AUDIT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    history_files = sorted(
        AUDIT_HISTORY_DIR.glob("*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )

    auditorias = []
    for history_file in history_files:
        try:
            with history_file.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
            payload["fonte"] = str(history_file)
            auditorias.append(payload)
        except Exception:
            continue

    if not auditorias and REPORT_PATH.exists():
        try:
            legacy_records = load_report_records(REPORT_PATH)
            if legacy_records:
                legacy_audit_id = legacy_records[0].get("auditoria_id") or "auditoria_legado"
                legacy_payload = {
                    "auditoria_id": legacy_audit_id,
                    "total_registros": len(legacy_records),
                    "registros": legacy_records,
                    "relatorio_path": str(REPORT_PATH),
                    "fonte": str(REPORT_PATH),
                }
                target_path = AUDIT_HISTORY_DIR / f"{legacy_audit_id}.json"
                if not target_path.exists():
                    with target_path.open("w", encoding="utf-8") as fp:
                        json.dump(legacy_payload, fp, ensure_ascii=False, indent=2)
                auditorias.append(legacy_payload)
        except Exception:
            pass

    return auditorias

# Configuração crucial do CORS para permitir que o React converse com o Python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # porta padrão do Vite
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint de teste para verificar se a API está funcionando
@app.get("/api/status")
@app.get("/status")
def read_status():
    return {"status": "AURA API is running!" , "version": "1.0"}

@app.post("/api/login")
def login(request: LoginRequest):
    """Verifica as credenciais enviadas pelo frontend."""
    from database import verify_user
    user = verify_user(request.username, request.password)
    
    if not user:
        raise HTTPException(status_code=401, detail="Credenciais Inválidas")
    
    # Retorna as info básicas ou pode-se gerar um token JWT
    return {"message": "Login successful", "user": user}

@app.get("/api/auditorias/ultima")
@app.get("/auditorias/ultima")
def get_latest_audit_result():
    """Retorna o último relatório de auditoria salvo para consumo do frontend."""
    if not REPORT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Nenhum relatório encontrado. Execute uma auditoria primeiro.",
        )

    try:
        records = load_report_records(REPORT_PATH)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Relatório corrompido: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Falha ao carregar relatório: {exc}",
        ) from exc

    if not records:
        raise HTTPException(
            status_code=404,
            detail="Relatório vazio. Execute uma auditoria primeiro.",
        )

    return {
        "auditoria_id": records[0].get("auditoria_id"),
        "total_registros": len(records),
        "registros": records,
        "fonte": str(REPORT_PATH),
    }


@app.get("/api/auditorias")
@app.get("/auditorias")
def get_all_audit_results(include_registros: bool = True):
    """Retorna todas as auditorias já realizadas (com opção de incluir registros)."""
    auditorias = []

    try:
        from database import get_all_assessments
        auditorias = get_all_assessments(include_records=include_registros)
    except Exception:
        auditorias = []

    if not auditorias:
        auditorias = load_audit_history_from_files()
        if not include_registros:
            auditorias = [
                {
                    "auditoria_id": item.get("auditoria_id"),
                    "total_registros": item.get("total_registros", 0),
                    "data_avaliacao": item.get("registros", [{}])[0].get("data_avaliacao") if item.get("registros") else None,
                    "fonte": item.get("fonte"),
                }
                for item in auditorias
            ]

    return {
        "total_auditorias": len(auditorias),
        "auditorias": auditorias,
    }

@app.post("/api/auditar/documentos")
@app.post("/auditar/documentos")
async def upload_documents(
    files: List[UploadFile] = File(...),
    rebuild_index: bool = True,
    analyze: bool = True,
):
    """Recebe documentos, atualiza o índice e retorna o JSON da auditoria."""
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    CORPUS_PATH.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for uploaded in files:
        suffix = Path(uploaded.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Arquivo '{uploaded.filename}' com formato não suportado.",
            )

        target_name = f"{uuid.uuid4()}{suffix}"
        target_path = CORPUS_PATH / target_name

        try:
            with target_path.open("wb") as buffer:
                shutil.copyfileobj(uploaded.file, buffer)
        finally:
            uploaded.file.close()

        saved_files.append(target_name)

    if rebuild_index:
        try:
            from ingestion import initialize_vector_store
            initialize_vector_store()
        except ModuleNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Dependência ausente para indexação vetorial. "
                    "Instale os pacotes de ingestão antes de usar rebuild_index=true."
                ),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Falha ao reconstruir índice vetorial: {exc}",
            ) from exc

    audit_payload = None
    if analyze:
        try:
            from main import execute_audit_process
            audit_payload = execute_audit_process(save_output=True, save_db=True)
            persist_audit_history(audit_payload)
        except ModuleNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Dependência ausente para executar a auditoria. "
                    "Instale os pacotes de IA necessários."
                ),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Falha ao executar auditoria: {exc}",
            ) from exc

    return {
        "message": "Documentos recebidos com sucesso.",
        "quantidade": len(saved_files),
        "arquivos": saved_files,
        "indice_atualizado": rebuild_index,
        "analise_executada": analyze,
        "resultado_auditoria": audit_payload,
    }

if __name__ == "__main__":
    import uvicorn
    
    # roda o servidor na porta 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)