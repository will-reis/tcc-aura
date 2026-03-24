"""
Módulo de Ingestão de Dados (RAG Pipeline).
Responsável por carregar documentos do corpus (PDF, TXT, DOCX), realizar chunking e popular o Vector Database.
"""
import os
import shutil
# Adicionamos o Docx2txtLoader na linha abaixo
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Configurações de Diretório
CORPUS_PATH = "corpus"
VECTOR_STORE_PATH = "vector_store"
MODEL_NAME = "llama3"

def initialize_vector_store():
    """Lê documentos e recria o índice vetorial local."""
    
    print(f"[INFO] Iniciando pipeline de ingestão de dados...")
    
    if not os.path.exists(CORPUS_PATH):
        os.makedirs(CORPUS_PATH)
        print(f"[WARN] Diretório '{CORPUS_PATH}' não encontrado. Criado automaticamente.")
        return

    # Pipeline de Carregamento (Agora com DOCX)
    loaders = {
        "PDF": DirectoryLoader(CORPUS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader),
        "TXT": DirectoryLoader(CORPUS_PATH, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'}),
        "DOCX": DirectoryLoader(CORPUS_PATH, glob="**/*.docx", loader_cls=Docx2txtLoader)
    }

    raw_documents = []
    for doc_type, loader in loaders.items():
        try:
            loaded = loader.load()
            print(f"[INFO] {len(loaded)} páginas/documentos do tipo {doc_type} carregados.")
            raw_documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Falha ao carregar {doc_type}: {e}")

    if not raw_documents:
        print("[ERROR] Corpus vazio. Abortando ingestão.")
        return

    # Estratégia de Chunking (Otimizada para contexto técnico)
    print("[INFO] Executando text splitting...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"[INFO] Total de fragmentos gerados: {len(chunks)}")

    # Geração de Embeddings
    print(f"[INFO] Gerando embeddings e persistindo no ChromaDB ({VECTOR_STORE_PATH})...")
    # Limpa banco anterior para evitar duplicidade
    if os.path.exists(VECTOR_STORE_PATH):
        shutil.rmtree(VECTOR_STORE_PATH)

    embedding_fn = OllamaEmbeddings(model=MODEL_NAME)
    
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_fn, 
        persist_directory=VECTOR_STORE_PATH
    )
    
    print(f"[SUCCESS] Base de conhecimento atualizada com sucesso no diretório '{VECTOR_STORE_PATH}'.")

if __name__ == "__main__":
    initialize_vector_store()