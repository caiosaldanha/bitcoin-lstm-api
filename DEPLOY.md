# 🚀 Deploy no Dokploy - Bitcoin LSTM Predictor

Este documento descreve como fazer o deploy da aplicação Bitcoin LSTM Predictor no Dokploy.

## 📋 Pré-requisitos

- Dokploy configurado em sua VPS
- Traefik configurado com certificados SSL
- Domínio configurado: `lstm.ml.caiosaldanha.com`

## 🔧 Configuração no Dokploy

### 1. Criar Novo Projeto

1. Acesse o painel do Dokploy
2. Clique em "New Project"
3. Nome: `bitcoin-lstm-predictor`
4. Conecte seu repositório GitHub

### 2. Configurar Variáveis de Ambiente

No painel do Dokploy, adicione as seguintes variáveis:

```env
ENVIRONMENT=production
DOMAIN=lstm.ml.caiosaldanha.com
PYTHONUNBUFFERED=1
TF_CPP_MIN_LOG_LEVEL=2
MODEL_EPOCHS=10
MODEL_BATCH_SIZE=1
GRAFANA_ADMIN_PASSWORD=seu_password_seguro
```

### 3. Configurar Build

- **Build Command**: `docker-compose build`
- **Dockerfile**: `Dockerfile`
- **Context**: `./`

### 4. Configurar Deploy

- **Deploy Command**: `docker-compose up -d`
- **Port**: `8000`
- **Health Check**: `/health`

## 🌐 URLs de Acesso

Após o deploy, a aplicação estará disponível em:

- **API Principal**: https://lstm.ml.caiosaldanha.com
- **Documentação**: https://lstm.ml.caiosaldanha.com/docs
- **Prometheus**: https://lstm.ml.caiosaldanha.com/prometheus
- **Grafana**: https://lstm.ml.caiosaldanha.com/grafana

## 🔐 Credenciais

### Grafana
- **Usuário**: admin
- **Senha**: Definida na variável `GRAFANA_ADMIN_PASSWORD`

## 📊 Monitoramento

A aplicação inclui monitoramento completo:

### Métricas Prometheus
Disponíveis em `/metrics`:
- Requisições por endpoint
- Tempo de resposta
- Uso de CPU/Memória
- Predições realizadas
- Acurácia do modelo

### Dashboard Grafana
Dashboard pré-configurado com:
- Taxa de requisições
- Tempo de resposta (percentis)
- Uso de recursos do sistema
- Performance do modelo ML
- Contadores de predições

## 🚀 Deploy Manual

Se preferir fazer deploy manual via SSH:

```bash
# 1. Conectar na VPS
ssh seu-usuario@sua-vps

# 2. Clonar repositório
git clone https://github.com/seu-usuario/bitcoin-lstm-predictor.git
cd bitcoin-lstm-predictor

# 3. Deploy
ENVIRONMENT=production DOMAIN=lstm.ml.caiosaldanha.com ./run.sh deploy
```

## 🔄 Comandos de Gerenciamento

```bash
# Verificar status
./run.sh status

# Ver logs
./run.sh logs

# Treinar modelo
./run.sh train

# Fazer predição
./run.sh predict

# Avaliar modelo
./run.sh evaluate

# Parar aplicação
./run.sh stop

# Reiniciar
./run.sh restart
```

## 🧪 Teste após Deploy

```bash
# Teste de saúde
curl https://lstm.ml.caiosaldanha.com/health

# Treinar modelo
curl -X POST https://lstm.ml.caiosaldanha.com/train

# Fazer predição
curl https://lstm.ml.caiosaldanha.com/predict

# Ver métricas
curl https://lstm.ml.caiosaldanha.com/monitoring
```

## 📝 Logs e Debug

### Ver logs via Dokploy
1. Acesse o painel do Dokploy
2. Vá em "Applications"
3. Selecione `bitcoin-lstm-predictor`  
4. Clique em "Logs"

### Via SSH
```bash
# Logs da aplicação
docker logs bitcoin-lstm-api -f

# Logs do Prometheus
docker logs bitcoin-lstm-prometheus -f

# Logs do Grafana
docker logs bitcoin-lstm-grafana -f
```

## 🔧 Troubleshooting

### Problema: Container não inicia
```bash
# Verificar status
docker ps -a

# Ver logs detalhados
docker logs bitcoin-lstm-api

# Rebuild clean
docker-compose down
docker system prune -f
docker-compose up -d --build
```

### Problema: SSL não funciona
- Verificar se domínio aponta para VPS
- Verificar configuração do Traefik
- Verificar labels do Docker Compose

### Problema: Grafana sem dados
```bash
# Verificar targets do Prometheus
curl http://localhost:9090/api/v1/targets

# Reiniciar stack
docker-compose restart
```

### Problema: Modelo não treina
- Verificar conectividade com Yahoo Finance
- Verificar recursos disponíveis (RAM/CPU)
- Ver logs específicos do erro

## 📊 Métricas de Performance

### Recursos Mínimos Recomendados
- **CPU**: 2 vCPUs
- **RAM**: 4GB  
- **Storage**: 10GB SSD
- **Largura de Banda**: 100Mbps

### Limites de Container
Configurados no docker-compose.yml:
- API: 2GB RAM, 1.5 CPU
- Prometheus: 1GB RAM, 0.5 CPU  
- Grafana: 512MB RAM, 0.3 CPU

## 🔄 Atualizações

Para atualizar a aplicação:

```bash
# Via Dokploy: Push para branch main
git push origin main

# Via SSH manual:
git pull origin main
docker-compose down
docker-compose up -d --build
```

## 🆘 Suporte

- **Issues**: GitHub Issues do projeto
- **Logs**: Disponíveis via Dokploy ou SSH
- **Monitoring**: Dashboard Grafana em tempo real
- **Health**: Endpoint `/health` para verificações
