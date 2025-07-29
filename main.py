from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import yfinance as yf
import joblib
import datetime
import os
import time
import psutil
import json
import logging
from typing import Dict, Any, List
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from contextlib import asynccontextmanager

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Métricas do Prometheus
REQUEST_COUNT = Counter('bitcoin_lstm_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('bitcoin_lstm_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
MODEL_PREDICTIONS = Counter('bitcoin_lstm_predictions_total', 'Total predictions made')
MODEL_TRAINING_TIME = Histogram('bitcoin_lstm_training_duration_seconds', 'Model training duration')
SYSTEM_CPU_USAGE = Gauge('bitcoin_lstm_cpu_usage_percent', 'CPU usage percentage')
SYSTEM_MEMORY_USAGE = Gauge('bitcoin_lstm_memory_usage_bytes', 'Memory usage in bytes')
MODEL_ACCURACY_GAUGE = Gauge('bitcoin_lstm_model_r2_score', 'Model R2 score from last training')

# Armazenar métricas do último treinamento
training_metrics_cache = {}

app = FastAPI(title="Bitcoin LSTM Predictor", version="1.0.0")

class TrainingResponse(BaseModel):
    message: str
    rmse: float
    mae: float
    r2: float
    model_saved: bool

class PredictionResponse(BaseModel):
    current_date: str
    next_day_prediction: float
    last_known_price: float

class ModelEvaluationResponse(BaseModel):
    model_exists: bool
    training_date: str
    rmse: float
    mae: float
    r2: float
    training_duration: float
    data_points_used: int

class MonitoringResponse(BaseModel):
    timestamp: str
    cpu_usage_percent: float
    memory_usage_mb: float
    total_requests: int
    total_predictions: int
    average_response_time_ms: float
    model_status: str

# Função para atualizar métricas do sistema
def update_system_metrics():
    """Atualiza métricas de CPU e memória"""
    SYSTEM_CPU_USAGE.set(psutil.cpu_percent())
    SYSTEM_MEMORY_USAGE.set(psutil.virtual_memory().used)

app = FastAPI(title="Bitcoin LSTM Predictor", version="1.0.0")

# Middleware para monitoramento
@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Atualizar métricas do sistema
    update_system_metrics()
    
    # Processar requisição
    response = await call_next(request)
    
    # Calcular duração
    duration = time.time() - start_time
    
    # Atualizar métricas
    method = request.method
    endpoint = request.url.path
    status = str(response.status_code)
    
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
    
    return response

def prepare_data():
    """Função para preparar os dados do Bitcoin"""
    ticker_symbol = "BTC-USD"
    start_date = "2018-01-01"
    end_date = datetime.date.today().strftime("%Y-%m-%d")

    # Obter dados do Yahoo Finance
    try:
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
    except Exception as e:
        if "No timezone found" in str(e):
            logger.error(f"Erro de fuso horário ao baixar dados para {ticker_symbol}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Erro de fuso horário para o símbolo {ticker_symbol}. O símbolo pode estar deslistado ou inválido.")
        logger.error(f"Erro ao baixar dados do Yahoo Finance: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro ao obter dados do Yahoo Finance. Verifique o símbolo ou a conectividade.")

    if data.empty:
        raise HTTPException(status_code=400, detail="Nenhum dado retornado para o período especificado. O símbolo pode estar deslistado ou inválido.")

    # Selecionar os preços de fechamento
    close_prices = data['Close'].values
    if len(close_prices) == 0:
        raise HTTPException(status_code=400, detail="Dados insuficientes para treinamento.")

    close_prices = close_prices.reshape(-1, 1)

    # Escalar os dados entre 0 e 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Definir o tamanho dos dados de treino
    training_data_len = int(np.ceil(len(scaled_data) * 0.8))

    # Dividir os dados
    train_data = scaled_data[:training_data_len]
    test_data = scaled_data[training_data_len - 40:]

    # Verificar se os preços de fechamento estão disponíveis
    if 'Close' not in data.columns or data['Close'].isnull().all():
        raise HTTPException(status_code=400, detail="Dados insuficientes ou ausentes para o período especificado.")

    return data, close_prices, scaler, scaled_data, train_data, test_data, training_data_len

def create_training_data(train_data):
    """Criar dataset de treino"""
    x_train, y_train = [], []
    
    for i in range(40, len(train_data)):
        x_train.append(train_data[i-40:i, 0])
        y_train.append(train_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train

def create_test_data(test_data, close_prices, training_data_len):
    """Criar dataset de teste"""
    x_test = []
    y_test = close_prices[training_data_len:]
    
    for i in range(40, len(test_data)):
        x_test.append(test_data[i-40:i, 0])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    return x_test, y_test

def build_lstm_model(input_shape):
    """Construir o modelo LSTM"""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {
        "message": "Bitcoin LSTM Predictor API", 
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.datetime.now().isoformat(),
        "model_status": "ready" if (
            os.path.exists('lstm_files/lstm_model.joblib') and 
            os.path.exists('lstm_files/scaler.joblib')
        ) else "needs_training"
    }

@app.post("/train", response_model=TrainingResponse)
async def train_model():
    """Rota para treinar o modelo LSTM e salvar em formato joblib"""
    start_time = time.time()
    
    try:
        # Aquisição dos dados
        ticker_symbol = "BTC-USD"
        start_date = "2018-01-01"
        end_date = datetime.date.today().strftime("%Y-%m-%d")
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        # Verificações robustas como no notebook
        if data is None or data.empty:
            raise HTTPException(status_code=400, detail="Nenhum dado retornado para o período especificado. O símbolo pode estar deslistado ou inválido.")
        if 'Close' not in data.columns or data['Close'].isnull().all():
            raise HTTPException(status_code=400, detail="Dados insuficientes ou ausentes para o período especificado.")
        close_prices = data['Close'].values
        if len(close_prices) == 0:
            raise HTTPException(status_code=400, detail="Dados insuficientes para treinamento.")
        close_prices = close_prices.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)
        training_data_len = int(np.ceil(len(scaled_data) * 0.8))
        train_data = scaled_data[:training_data_len]
        test_data = scaled_data[training_data_len - 40:]
        # Criação dos dados de treino
        x_train, y_train = [], []
        for i in range(40, len(train_data)):
            x_train.append(train_data[i-40:i, 0])
            y_train.append(train_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        # Construção do modelo
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        # Treinamento
        history = model.fit(x_train, y_train, validation_split=0.2, batch_size=1, epochs=10, verbose=0)
        # Criação dos dados de teste
        x_test = []
        y_test = close_prices[training_data_len:]
        for i in range(40, len(test_data)):
            x_test.append(test_data[i-40:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        # Predição
        predictions = model.predict(x_test, verbose=0)
        predictions = scaler.inverse_transform(predictions)
        # Avaliação
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        valid_data = data[training_data_len:].copy()
        y_test_actual = valid_data['Close'].values
        min_len = min(len(y_test_actual), len(predictions))
        y_test_actual = y_test_actual[:min_len]
        predictions_adjusted = predictions[:min_len].flatten()
        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_adjusted))
        mae = mean_absolute_error(y_test_actual, predictions_adjusted)
        r2 = r2_score(y_test_actual, predictions_adjusted)
        training_duration = time.time() - start_time
        # Salvar modelo e scaler
        if not os.path.exists('lstm_files'):
            os.makedirs('lstm_files')
        joblib.dump(model, 'lstm_files/lstm_model.joblib')
        joblib.dump(scaler, 'lstm_files/scaler.joblib')
        # Armazenar métricas no cache
        training_metrics_cache.update({
            'training_date': datetime.datetime.now().isoformat(),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'training_duration': training_duration,
            'data_points_used': len(close_prices)
        })
        with open('lstm_files/training_metrics.json', 'w') as f:
            json.dump(training_metrics_cache, f, indent=2)
        MODEL_TRAINING_TIME.observe(training_duration)
        MODEL_ACCURACY_GAUGE.set(r2)
        return TrainingResponse(
            message="Modelo treinado e salvo com sucesso!",
            rmse=float(rmse),
            mae=float(mae),
            r2=float(r2),
            model_saved=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no treinamento: {str(e)}")

@app.get("/predict", response_model=PredictionResponse)
async def predict_next_day():
    """Rota para predição do próximo dia"""
    try:
        # Verificar se os arquivos do modelo existem
        if not os.path.exists('lstm_files/lstm_model.joblib') or not os.path.exists('lstm_files/scaler.joblib'):
            raise HTTPException(status_code=404, detail="Modelo não encontrado. Execute o treinamento primeiro.")
        
        # Carregar modelo e scaler
        model = joblib.load('lstm_files/lstm_model.joblib')
        scaler = joblib.load('lstm_files/scaler.joblib')
        
        # Obter dados mais recentes
        ticker_symbol = "BTC-USD"
        end_date = datetime.date.today().strftime("%Y-%m-%d")
        start_date = (datetime.date.today() - datetime.timedelta(days=100)).strftime("%Y-%m-%d")
        
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        
        if len(data) < 40:
            raise HTTPException(status_code=400, detail="Dados insuficientes para predição")
        
        # Preparar dados para predição
        close_prices = data['Close'].values
        close_prices = close_prices.reshape(-1, 1)
        
        # Escalar dados
        scaled_data = scaler.transform(close_prices)
        
        # Pegar os últimos 40 dias
        last_40_days = scaled_data[-40:]
        last_40_days = last_40_days.reshape(1, 40, 1)
        
        # Fazer predição
        prediction = model.predict(last_40_days, verbose=0)
        prediction = scaler.inverse_transform(prediction)
        
        # Incrementar contador de predições
        MODEL_PREDICTIONS.inc()
        
        # Obter último preço conhecido
        last_known_price = float(close_prices[-1][0])
        predicted_price = float(prediction[0][0])
        
        current_date = datetime.date.today().strftime("%Y-%m-%d")
        
        return PredictionResponse(
            current_date=current_date,
            next_day_prediction=predicted_price,
            last_known_price=last_known_price
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Informações sobre o modelo"""
    model_exists = os.path.exists('lstm_files/lstm_model.joblib')
    scaler_exists = os.path.exists('lstm_files/scaler.joblib')
    
    return {
        "model_exists": model_exists,
        "scaler_exists": scaler_exists,
        "model_path": "lstm_files/lstm_model.joblib",
        "scaler_path": "lstm_files/scaler.joblib",
        "status": "ready" if (model_exists and scaler_exists) else "needs_training"
    }

@app.get("/evaluate", response_model=ModelEvaluationResponse)
async def evaluate_model():
    """Rota para avaliar o modelo com métricas da fase de treino"""
    try:
        # Verificar se o arquivo de métricas existe
        if not os.path.exists('lstm_files/training_metrics.json'):
            raise HTTPException(status_code=404, detail="Métricas de treinamento não encontradas. Execute o treinamento primeiro.")
        
        # Carregar métricas do arquivo
        with open('lstm_files/training_metrics.json', 'r') as f:
            metrics = json.load(f)
        
        return ModelEvaluationResponse(
            model_exists=os.path.exists('lstm_files/lstm_model.joblib'),
            training_date=metrics.get('training_date', 'Unknown'),
            rmse=metrics.get('rmse', 0.0),
            mae=metrics.get('mae', 0.0),
            r2=metrics.get('r2', 0.0),
            training_duration=metrics.get('training_duration', 0.0),
            data_points_used=metrics.get('data_points_used', 0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na avaliação: {str(e)}")

@app.get("/monitoring", response_model=MonitoringResponse)
async def get_monitoring_info():
    """Rota para obter informações de monitoramento"""
    try:
        # Atualizar métricas do sistema
        update_system_metrics()
        
        # Calcular estatísticas das requisições
        total_requests = sum([
            metric.samples[0].value for metric in REQUEST_COUNT.collect()
            if metric.samples
        ])
        
        total_predictions = sum([
            metric.samples[0].value for metric in MODEL_PREDICTIONS.collect()
            if metric.samples
        ])
        
        # Calcular tempo médio de resposta
        duration_samples = []
        for metric in REQUEST_DURATION.collect():
            for sample in metric.samples:
                if sample.name.endswith('_sum'):
                    duration_samples.append(sample.value)
        
        avg_response_time = (sum(duration_samples) / len(duration_samples) * 1000) if duration_samples else 0.0
        
        # Status do modelo
        model_status = "ready" if (
            os.path.exists('lstm_files/lstm_model.joblib') and 
            os.path.exists('lstm_files/scaler.joblib')
        ) else "needs_training"
        
        return MonitoringResponse(
            timestamp=datetime.datetime.now().isoformat(),
            cpu_usage_percent=psutil.cpu_percent(),
            memory_usage_mb=psutil.virtual_memory().used / 1024 / 1024,
            total_requests=int(total_requests),
            total_predictions=int(total_predictions),
            average_response_time_ms=avg_response_time,
            model_status=model_status
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no monitoramento: {str(e)}")

@app.get("/metrics")
async def get_prometheus_metrics():
    """Endpoint para métricas do Prometheus"""
    update_system_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check endpoint accessed")
    
    try:
        # Verificar saúde básica da aplicação
        model_exists = os.path.exists('lstm_files/lstm_model.joblib')
        scaler_exists = os.path.exists('lstm_files/scaler.joblib')
        
        # Verificar recursos do sistema
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "version": "1.0.0",
            "model_ready": model_exists and scaler_exists,
            "system": {
                "cpu_usage_percent": cpu_usage,
                "memory_usage_percent": memory.percent,
                "memory_available_mb": memory.available / 1024 / 1024
            },
            "components": {
                "model_file": model_exists,
                "scaler_file": scaler_exists,
                "lstm_files_dir": os.path.exists('lstm_files')
            }
        }
        
        logger.info(f"Health check result: {health_status}")
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
