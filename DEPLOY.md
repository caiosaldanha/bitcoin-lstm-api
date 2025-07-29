# 🚀 Deploy no Dokploy - Bitcoin LSTM Predictor

Este documento descreve como fazer o deploy da aplicação Bitcoin LSTM Predictor no **Dokploy** com monitoramento completo.

## 📋 Pré-requisitos

- Dokploy configurado em sua VPS
- Docker e Docker Compose instalados
- Domínio configurado: `lstm.ml.caiosaldanha.com`
- Repositório no GitHub

## 🔧 Configuração no Dokploy

### 1. Conectar Repositório GitHub

1. Acesse o painel do Dokploy
2. Clique em "New Project"
3. Nome: `bitcoin-lstm-predictor`
4. Conecte seu repositório GitHub
5. Branch: `main`
6. Configure para deploy automático

### 2. Configuração de Deploy

#### Configurações Básicas:
- **Build Context**: `./`
- **Dockerfile**: `Dockerfile`
- **Docker Compose File**: `docker-compose.yml`
- **Port**: `8000`
- **Health Check Path**: `/health`

#### Variáveis de Ambiente (Opcionais):
```env
ENVIRONMENT=production
PYTHONUNBUFFERED=1
TF_CPP_MIN_LOG_LEVEL=2
GF_SECURITY_ADMIN_PASSWORD=seu_password_seguro
```

## 🌐 URLs de Acesso

Após o deploy, a aplicação estará disponível em:

- **API Principal**: https://lstm.ml.caiosaldanha.com
- **Documentação Swagger**: https://lstm.ml.caiosaldanha.com/docs
- **Health Check**: https://lstm.ml.caiosaldanha.com/health
- **Prometheus**: https://lstm.ml.caiosaldanha.com/prometheus
- **Grafana**: https://lstm.ml.caiosaldanha.com/grafana

### Credenciais Grafana
- **Usuário**: `admin`
- **Senha**: `admin123` (padrão) ou conforme configurado na variável de ambiente

## ⚙️ Stack Completa

O projeto inclui três serviços principais:

### 1. Bitcoin LSTM API (Porta 8000)
- FastAPI com endpoints de treino e predição
- Métricas Prometheus integradas
- Health checks automáticos
- Persistência de modelos

### 2. Prometheus (Porta 9090)
- Coleta de métricas da API
- Retenção de dados por 200h
- Interface web para consultas
- Configuração automática

### 3. Grafana (Porta 3000)
- Dashboard pré-configurado
- Visualizações em tempo real
- Métricas de performance
- Alertas personalizáveis

## ⚙️ Configuração Automática

O `docker-compose.yml` inclui todos os serviços:

```yaml
services:
  app:
    # API principal com métricas
    container_name: bitcoin-lstm-api
    labels:
      - traefik.http.routers.lstm-api.rule=Host(`lstm.ml.caiosaldanha.com`)
      
  prometheus:
    # Monitoramento de métricas
    container_name: bitcoin-lstm-prometheus
    labels:
      - traefik.http.routers.lstm-prometheus.rule=Host(`lstm.ml.caiosaldanha.com`) && PathPrefix(`/prometheus`)
      
  grafana:
    # Dashboards e visualizações
    container_name: bitcoin-lstm-grafana
    labels:
      - traefik.http.routers.lstm-grafana.rule=Host(`lstm.ml.caiosaldanha.com`) && PathPrefix(`/grafana`)
```

## ✅ Verificações Pós-Deploy

### 1. Health Check da API
```bash
curl https://lstm.ml.caiosaldanha.com/health
```

### 2. Métricas Prometheus
```bash
curl https://lstm.ml.caiosaldanha.com/metrics
```

### 3. Acesso ao Grafana
```bash
# Acesse no navegador
https://lstm.ml.caiosaldanha.com/grafana
```

### 4. Teste Completo
```bash
# Treinar modelo
curl -X POST "https://lstm.ml.caiosaldanha.com/train"

# Fazer predição
curl -X POST "https://lstm.ml.caiosaldanha.com/predict"

# Ver métricas
curl "https://lstm.ml.caiosaldanha.com/monitoring"
```

## 📊 Monitoramento Disponível

### Métricas da API:
- **Taxa de requisições** por endpoint
- **Tempo de resposta** (percentis 50, 95, 99)
- **Número de predições** realizadas
- **Acurácia do modelo** em tempo real
- **Uso de CPU e memória**

### Dashboard Grafana:
- **Request Rate**: Taxa de requisições em tempo real
- **Response Time**: Latência da API
- **Total Predictions**: Contador de predições
- **System Resources**: Uso de recursos
- **Model Performance**: Métricas do LSTM

## 🔄 Deploy Automático

O projeto está configurado para deploy automático:

1. **Push para main**: Cada commit triggers novo deploy
2. **Build multi-serviço**: API, Prometheus e Grafana
3. **Health checks**: Verificação automática dos serviços
4. **SSL automático**: Let's Encrypt para todos os serviços
5. **Persistência**: Volumes para modelos e dados de monitoramento

## 🔧 Troubleshooting

### Problema: Grafana não carrega
**Solução**: Verifique se o serviço está rodando e se o path `/grafana` está acessível:
```bash
docker-compose logs grafana
```

### Problema: Métricas não aparecem
**Solução**: Verifique se o Prometheus está coletando dados da API:
```bash
# Acesse Prometheus e verifique targets
https://lstm.ml.caiosaldanha.com/prometheus/targets
```

### Problema: Modelo não encontrado
**Solução**: Execute o treinamento inicial:
```bash
curl -X POST "https://lstm.ml.caiosaldanha.com/train"
```

## 🔒 Segurança

- ✅ HTTPS automático via Let's Encrypt
- ✅ Containers não-root
- ✅ Health checks integrados
- ✅ Restart automático em caso de falha
- ✅ Volumes persistentes isolados
- ✅ Rede Docker isolada (dokploy-network)

## 📝 Logs

Para visualizar logs dos serviços:

```bash
# Logs da API
docker-compose logs -f app

# Logs do Prometheus
docker-compose logs -f prometheus

# Logs do Grafana
docker-compose logs -f grafana
```

## 🎯 Próximos Passos

Após o deploy bem-sucedido:

1. **Execute o treinamento inicial**:
   ```bash
   curl -X POST "https://lstm.ml.caiosaldanha.com/train"
   ```

2. **Acesse o Grafana** e configure alertas se necessário

3. **Monitore performance** via dashboard

4. **Configure backup** dos volumes de dados se necessário

---

✨ **Deploy completo com monitoramento integrado!**
