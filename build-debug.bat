@echo off
echo === Verificando arquivos antes do build Docker ===
echo Conteudo atual do diretorio lstm_files:
dir lstm_files

echo.
echo === Iniciando build Docker ===
docker build -t bitcoin-lstm-api-debug .

echo.
echo === Build concluido ===
pause
