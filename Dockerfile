FROM python:3.11-slim

# Definir variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema mínimas
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copiar arquivos de requisitos
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY lstm_files lstm_files

# Debug: List contents of lstm_files to verify copy
RUN echo "=== Conteúdo do diretório lstm_files após COPY ===" && \
    ls -la lstm_files/ && \
    echo "=== Verificando tamanhos dos arquivos ===" && \
    find lstm_files/ -name "*.joblib" -exec ls -lh {} \; && \
    echo "=== Fim da verificação ==="

# Garantir permissões corretas e checar arquivos obrigatórios
RUN chmod -R 755 lstm_files && \
    test -f lstm_files/lstm_model.joblib && \
    test -f lstm_files/scaler.joblib || (echo "ERRO: lstm_model.joblib ou scaler.joblib não encontrados em lstm_files. Adicione os arquivos antes do build." && exit 1)

# Criar usuário não-root para segurança
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expor porta
EXPOSE 8000

# Comando para executar a aplicação
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
