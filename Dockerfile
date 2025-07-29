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
# Garantir permissões corretas
RUN chmod -R 755 lstm_files

# Criar usuário não-root para segurança
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expor porta
EXPOSE 8000

# Comando para executar a aplicação
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
