CREATE DATABASE nist_audit_db;
USE nist_audit_db;
-- Tabela de usuários para o Login
CREATE TABLE usuarios (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'consultor'
);

-- Tabela para agrupar as auditorias (Lotes)
CREATE TABLE assessments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    auditoria_uuid VARCHAR(50) UNIQUE NOT NULL,
    usuario_id INT,
    data_avaliacao DATETIME,
    FOREIGN KEY (usuario_id) REFERENCES usuarios(id)
);

-- Tabela para os resultados detalhados (Cada linha do seu JSONL)
CREATE TABLE resultados_nist (
    id INT AUTO_INCREMENT PRIMARY KEY,
    assessment_id INT,
    registro_uuid VARCHAR(50),
    funcao VARCHAR(100),
    categoria VARCHAR(200),
    subcategoria_id VARCHAR(50),
    descricao TEXT,
    nivel_maturidade VARCHAR(100),
    pontuacao INT,
    cenario_atual TEXT,
    evidencia TEXT,
    gaps TEXT,
    recomendacoes JSON, -- Armazena a lista como JSON no MySQL
    FOREIGN KEY (assessment_id) REFERENCES assessments(id)
);
