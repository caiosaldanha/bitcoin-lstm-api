import os
import joblib
import datetime
import numpy as np
import yfinance as yf
import time
import traceback
import json
from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Função para carregar modelo com compatibilidade de versões
def load_model_safely(model_path):
    """Carrega o modelo de forma segura, lidando com diferentes versões e formatos"""
    import os
    
    # Primeiro verifica se é um modelo Keras salvo no formato nativo
    keras_path = model_path.replace('.joblib', '.keras')
    h5_path = model_path.replace('.joblib', '.h5')
    
    try:
        # Tentativa 1: Carregar como modelo Keras nativo
        if os.path.exists(keras_path):
            import tensorflow as tf
            return tf.keras.models.load_model(keras_path)
        elif os.path.exists(h5_path):
            import tensorflow as tf
            return tf.keras.models.load_model(h5_path)
    except Exception as e:
        print(f"Falha ao carregar modelo Keras nativo: {e}")
    
    try:
        # Tentativa 2: Carregamento joblib normal
        return joblib.load(model_path)
    except (ImportError, ModuleNotFoundError) as e:
        if 'keras' in str(e):
            try:
                # Tentativa 3: Com tensorflow.keras
                import tensorflow as tf
                import sys
                if 'keras.src' in str(e):
                    # Redirect keras imports to tensorflow.keras
                    sys.modules['keras.src'] = tf.keras
                    sys.modules['keras.src.models'] = tf.keras.models
                    sys.modules['keras.src.models.sequential'] = tf.keras.models
                
                return joblib.load(model_path)
            except Exception as e2:
                raise Exception(f"Erro ao carregar modelo: {str(e)}. Tentativa alternativa falhou: {str(e2)}")
        else:
            raise e
    except AttributeError as e:
        if '_unpickle_model' in str(e):
            try:
                # Tentativa 4: Usar tensorflow.keras.models.load_model com joblib 
                import tensorflow as tf
                import pickle
                
                # Tenta carregar diretamente como pickle
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e2:
                raise Exception(f"Modelo foi salvo incorretamente com joblib. Erro original: {str(e)}. Erro pickle: {str(e2)}. Recomenda-se retreinar o modelo usando model.save() do Keras.")
        else:
            raise e


# Pydantic Models
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


# Prometheus Metrics
REQUEST_COUNT = Counter('bitcoin_lstm_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('bitcoin_lstm_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
MODEL_STATUS = Gauge('bitcoin_lstm_model_status', 'Model status: 1=ready, 0=needs_training')
PREDICTION_COUNT = Counter('bitcoin_lstm_prediction_total', 'Total predictions made')
ERROR_COUNT = Counter('bitcoin_lstm_error_total', 'Total errors', ['endpoint'])


# FastAPI App
app = FastAPI(title="Bitcoin LSTM Predictor", version="1.0.0")


# Middleware para métricas Prometheus
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        status = str(response.status_code)
    except Exception:
        status = "500"
        raise
    duration = time.time() - start_time
    method = request.method
    endpoint = request.url.path
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
    return response


# Routes
@app.get("/")
async def root():
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


# Rota para checagem detalhada dos arquivos do modelo
@app.get("/model-check")
async def model_check() -> Dict:
    """Verifica a existência dos arquivos do modelo e retorna logs detalhados"""
    logs = []
    model_path = 'lstm_files/lstm_model.joblib'
    scaler_path = 'lstm_files/scaler.joblib'
    
    # Adiciona informações de diagnóstico
    logs.append(f"Diretório de trabalho atual: {os.getcwd()}")
    logs.append(f"Diretório lstm_files existe: {os.path.exists('lstm_files')}")
    
    # Lista conteúdo do diretório se existir
    if os.path.exists('lstm_files'):
        try:
            files_in_dir = os.listdir('lstm_files')
            logs.append(f"Arquivos em lstm_files: {files_in_dir}")
        except Exception as e:
            logs.append(f"Erro ao listar lstm_files: {str(e)}")
    
    result = {
        "model_exists": False,
        "scaler_exists": False,
        "model_path": os.path.abspath(model_path),
        "scaler_path": os.path.abspath(scaler_path),
        "working_directory": os.getcwd(),
        "details": []
    }
    
    try:
        if os.path.exists(model_path):
            result["model_exists"] = True
            file_size = os.path.getsize(model_path)
            logs.append(f"Arquivo encontrado: {model_path} (tamanho: {file_size} bytes)")
        else:
            logs.append(f"Arquivo NÃO encontrado: {model_path}")
            logs.append(f"Caminho absoluto: {os.path.abspath(model_path)}")

        if os.path.exists(scaler_path):
            result["scaler_exists"] = True
            file_size = os.path.getsize(scaler_path)
            logs.append(f"Arquivo encontrado: {scaler_path} (tamanho: {file_size} bytes)")
        else:
            logs.append(f"Arquivo NÃO encontrado: {scaler_path}")
            logs.append(f"Caminho absoluto: {os.path.abspath(scaler_path)}")

        # Tenta carregar os arquivos se existirem
        if result["model_exists"]:
            try:
                model = load_model_safely(model_path)
                logs.append(f"load_model_safely OK para {model_path}")
                logs.append(f"Tipo do modelo: {type(model)}")
            except Exception as e:
                logs.append(f"Erro ao carregar {model_path}: {str(e)}")
                
        if result["scaler_exists"]:
            try:
                scaler = joblib.load(scaler_path)
                logs.append(f"joblib.load OK para {scaler_path}")
                logs.append(f"Tipo do scaler: {type(scaler)}")
            except Exception as e:
                logs.append(f"Erro ao carregar {scaler_path}: {str(e)}")
                
    except Exception as e:
        logs.append(f"Erro geral na checagem: {str(e)}")
        logs.append(f"Traceback: {traceback.format_exc()}")
        
    result["details"] = logs
    return result


@app.get("/predict", response_model=PredictionResponse)
async def predict_next_day():
    """Rota para predição do próximo dia"""
    try:
        if not os.path.exists('lstm_files/lstm_model.joblib') or not os.path.exists('lstm_files/scaler.joblib'):
            ERROR_COUNT.labels(endpoint="/predict").inc()
            raise HTTPException(status_code=404, detail="Modelo não encontrado. Execute o treinamento primeiro.")

        model = load_model_safely('lstm_files/lstm_model.joblib')
        scaler = joblib.load('lstm_files/scaler.joblib')

        ticker_symbol = "BTC-USD"
        end_date = datetime.date.today().strftime("%Y-%m-%d")
        start_date = (datetime.date.today() - datetime.timedelta(days=100)).strftime("%Y-%m-%d")

        data = yf.download(ticker_symbol, start=start_date, end=end_date)

        if len(data) < 40:
            ERROR_COUNT.labels(endpoint="/predict").inc()
            raise HTTPException(status_code=400, detail="Dados insuficientes para predição")

        close_prices = data['Close'].values
        close_prices = close_prices.reshape(-1, 1)
        scaled_data = scaler.transform(close_prices)
        last_40_days = scaled_data[-40:]
        last_40_days = last_40_days.reshape(1, 40, 1)
        prediction = model.predict(last_40_days, verbose=0)
        prediction = scaler.inverse_transform(prediction)

        last_known_price = float(close_prices[-1][0])
        predicted_price = float(prediction[0][0])
        current_date = datetime.date.today().strftime("%Y-%m-%d")

        PREDICTION_COUNT.inc()

        return PredictionResponse(
            current_date=current_date,
            next_day_prediction=predicted_price,
            last_known_price=last_known_price
        )

    except Exception as e:
        ERROR_COUNT.labels(endpoint="/predict").inc()
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}\nTraceback:\n{tb}")

@app.get("/model-info")
async def model_info():
    """Informações sobre o modelo"""
    model_exists = os.path.exists('lstm_files/lstm_model.joblib')
    scaler_exists = os.path.exists('lstm_files/scaler.joblib')
    
    # Verifica formatos alternativos
    keras_exists = os.path.exists('lstm_files/lstm_model.keras')
    h5_exists = os.path.exists('lstm_files/lstm_model.h5')
    
    return {
        "model_exists": model_exists,
        "scaler_exists": scaler_exists,
        "keras_model_exists": keras_exists,
        "h5_model_exists": h5_exists,
        "model_path": "lstm_files/lstm_model.joblib",
        "scaler_path": "lstm_files/scaler.joblib",
        "status": "ready" if (model_exists and scaler_exists) else "needs_training",
        "recommendation": "Se o modelo não carrega, considere retreinar usando model.save() ao invés de joblib.dump()"
    }

@app.get("/evaluate", response_model=ModelEvaluationResponse)
async def evaluate_model():
    """Rota para avaliar o modelo com métricas da fase de treino"""
    try:
        if not os.path.exists('lstm_files/training_metrics.json'):
            ERROR_COUNT.labels(endpoint="/evaluate").inc()
            raise HTTPException(status_code=404, detail="Métricas de treinamento não encontradas. Execute o treinamento primeiro.")

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
        ERROR_COUNT.labels(endpoint="/evaluate").inc()
        raise HTTPException(status_code=500, detail=f"Erro na avaliação: {str(e)}")

@app.get("/monitoring", response_model=MonitoringResponse)
async def get_monitoring_info():
    """Rota para obter informações de monitoramento"""
    try:
        total_requests = sum([
            metric.samples[0].value for metric in REQUEST_COUNT.collect()
            if metric.samples
        ])
        total_predictions = PREDICTION_COUNT._value.get() if hasattr(PREDICTION_COUNT, '_value') else 0
        error_count = sum([
            sample.value for metric in ERROR_COUNT.collect() for sample in metric.samples if sample.name == 'bitcoin_lstm_error_total'])

        duration_samples = []
        for metric in REQUEST_DURATION.collect():
            for sample in metric.samples:
                if sample.name.endswith('_sum'):
                    duration_samples.append(sample.value)

        avg_response_time = (sum(duration_samples) / len(duration_samples) * 1000) if duration_samples else 0.0

        model_status = "ready" if (
            os.path.exists('lstm_files/lstm_model.joblib') and 
            os.path.exists('lstm_files/scaler.joblib')
        ) else "needs_training"

        return MonitoringResponse(
            timestamp=datetime.datetime.now().isoformat(),
            cpu_usage_percent=0.0,  # Not implemented
            memory_usage_mb=0.0,    # Not implemented
            total_requests=int(total_requests),
            total_predictions=int(total_predictions),
            average_response_time_ms=avg_response_time,
            model_status=model_status
        )

    except Exception as e:
        ERROR_COUNT.labels(endpoint="/monitoring").inc()
        raise HTTPException(status_code=500, detail=f"Erro no monitoramento: {str(e)}")


@app.get("/metrics")
async def get_prometheus_metrics():
    """Endpoint para métricas do Prometheus"""
    # Atualiza status do modelo
    model_ready = os.path.exists('lstm_files/lstm_model.joblib') and os.path.exists('lstm_files/scaler.joblib')
    MODEL_STATUS.set(1 if model_ready else 0)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Verificar saúde básica da aplicação
        model_exists = os.path.exists('lstm_files/lstm_model.joblib')
        scaler_exists = os.path.exists('lstm_files/scaler.joblib')

        health_status = {
            "status": "healthy" if (model_exists and scaler_exists) else "unhealthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "version": "1.0.0",
            "model_ready": model_exists and scaler_exists,
            "components": {
                "model_file": model_exists,
                "scaler_file": scaler_exists,
                "lstm_files_dir": os.path.exists('lstm_files')
            }
        }
        return health_status

    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
