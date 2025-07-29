#!/bin/bash

echo "=========================================="
echo "PRÉ-BUILD: Verificação dos arquivos modelo"
echo "=========================================="

echo "1. Verificando arquivos locais:"
if [ -f "lstm_files/lstm_model.joblib" ]; then
    echo "   ✅ lstm_model.joblib encontrado ($(ls -lh lstm_files/lstm_model.joblib | awk '{print $5}'))"
else
    echo "   ❌ lstm_model.joblib NÃO encontrado"
    exit 1
fi

if [ -f "lstm_files/scaler.joblib" ]; then
    echo "   ✅ scaler.joblib encontrado ($(ls -lh lstm_files/scaler.joblib | awk '{print $5}'))"
else
    echo "   ❌ scaler.joblib NÃO encontrado" 
    exit 1
fi

echo ""
echo "2. Verificando .dockerignore:"
if grep -q "^\*.joblib" .dockerignore; then
    echo "   ❌ PROBLEMA: .dockerignore está excluindo arquivos .joblib!"
    exit 1
else
    echo "   ✅ .dockerignore não exclui arquivos .joblib"
fi

echo ""
echo "3. Listando conteúdo completo de lstm_files/:"
ls -la lstm_files/

echo ""
echo "=========================================="
echo "✅ Verificação concluída - OK para build"
echo "=========================================="
