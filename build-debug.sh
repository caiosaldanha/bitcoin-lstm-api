#!/bin/bash

echo "=== Verificando arquivos antes do build Docker ==="
echo "Conteúdo atual do diretório lstm_files:"
ls -la lstm_files/

echo ""
echo "Tamanhos dos arquivos modelo:"
find lstm_files/ -name "*.joblib" -exec ls -lh {} \;

echo ""
echo "=== Iniciando build Docker ==="
docker build -t bitcoin-lstm-api-debug .

echo ""
echo "=== Build concluído ==="
