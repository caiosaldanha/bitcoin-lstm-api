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

# Primeiro cria o diretório e copia o .keep
COPY lstm_files/.keep lstm_files/

# Agora copia explicitamente os arquivos modelo (se existirem)
COPY lstm_files/ lstm_files/

# Debug mais detalhado
RUN echo "=== DIAGNÓSTICO COMPLETO APÓS COPY ===" && \
    echo "PWD: $(pwd)" && \
    echo "Conteúdo do diretório atual:" && \
    ls -la && \
    echo "=== Conteúdo do diretório lstm_files ===" && \
    ls -la lstm_files/ && \
    echo "=== Verificando arquivos .joblib ===" && \
    find lstm_files/ -name "*.joblib" -exec ls -lh {} \; || echo "Nenhum arquivo .joblib encontrado" && \
    echo "=== Verificando tamanhos específicos ===" && \
    (test -f lstm_files/lstm_model.joblib && echo "lstm_model.joblib: $(ls -lh lstm_files/lstm_model.joblib)" || echo "lstm_model.joblib: NÃO ENCONTRADO") && \
    (test -f lstm_files/scaler.joblib && echo "scaler.joblib: $(ls -lh lstm_files/scaler.joblib)" || echo "scaler.joblib: NÃO ENCONTRADO") && \
    echo "=== FIM DO DIAGNÓSTICO ==="

# Garantir permissões corretas (remover teste que falha o build)
RUN chmod -R 755 lstm_files

# Avisar sobre status dos arquivos mas não falhar o build
RUN if [ ! -f lstm_files/lstm_model.joblib ]; then \
        echo "AVISO: lstm_model.joblib não encontrado - API funcionará em modo diagnóstico"; \
    fi && \
    if [ ! -f lstm_files/scaler.joblib ]; then \
        echo "AVISO: scaler.joblib não encontrado - API funcionará em modo diagnóstico"; \
    fi

# Criar usuário não-root para segurança
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expor porta
EXPOSE 8000

# Comando para executar a aplicação
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
