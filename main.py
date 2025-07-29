import os
import joblib
import datetime
import numpy as np
import yfinance as yf
import time
import traceback
import json
from typing import Dict
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response


def fetch_bitcoin_data(start_date: str, end_date: str, cache_hours: int = 12):
    """Baixa dados do Bitcoin com cache e fonte alternativa."""
    cache_dir = "data"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"btc_{start_date}_{end_date}.csv")

    # Verifica cache existente e recente
    if os.path.exists(cache_file):
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.datetime.now() - mtime < datetime.timedelta(hours=cache_hours):
            try:
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
            except Exception as e:
                print(f"⚠️ Falha ao ler cache {cache_file}: {e}")

    # Tentativa principal com yfinance
    try:
        data = yf.download(
            "BTC-USD",
            start=start_date,
            end=end_date,
            progress=False,
            show_errors=False,
            threads=True,
        )
        if data is not None and len(data) > 0:
            data.to_csv(cache_file)
            return data
    except Exception as e:
        print(f"⚠️ Erro ao baixar com yfinance: {e}")

    # Fonte alternativa: Stooq
    try:
        url = "https://stooq.com/q/d/l/?s=btcusd&i=d"
        alt = pd.read_csv(url)
        alt.columns = ["Date", "Open", "High", "Low", "Close"]
        alt["Date"] = pd.to_datetime(alt["Date"])
        alt.set_index("Date", inplace=True)
        data = alt.loc[start_date:end_date]
        if len(data) > 0:
            data.to_csv(cache_file)
            return data
    except Exception as e:
        print(f"❌ Erro ao baixar dados da Stooq: {e}")

    return None


def create_compatible_model(sequence_length=40):
    """Cria um modelo LSTM compatível com diferentes versões do Keras"""
    import tensorflow as tf
    import os
    import json
    
    # Tenta carregar configuração salva
    config_path = 'lstm_files/model_config.json'
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            sequence_length = config.get('sequence_length', sequence_length)
            print(f"✅ Usando configuração salva: sequence_length={sequence_length}")
        except Exception as e:
            print(f"⚠️ Erro ao carregar configuração: {e}. Usando padrão.")
    
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, 1),
                           kernel_initializer='glorot_uniform',
                           recurrent_initializer='orthogonal'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50, return_sequences=True,
                           kernel_initializer='glorot_uniform',
                           recurrent_initializer='orthogonal'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50,
                           kernel_initializer='glorot_uniform',
                           recurrent_initializer='orthogonal'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, kernel_initializer='glorot_uniform')
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Função para carregar modelo com compatibilidade de versões
def load_model_safely(model_path):
    """Carrega o modelo de forma segura, priorizando formato .keras com fallbacks robustos"""
    import os
    import tensorflow as tf
    
    # Lista de possíveis caminhos na ordem de prioridade
    keras_path = model_path.replace('.joblib', '.keras').replace('.h5', '.keras')
    h5_path = model_path.replace('.joblib', '.h5').replace('.keras', '.h5')
    joblib_path = model_path.replace('.keras', '.joblib').replace('.h5', '.joblib')
    
    # Tentativa 1: Carregar modelo .keras (prioridade)
    if os.path.exists(keras_path):
        try:
            print(f"✅ Tentando carregar modelo Keras nativo: {keras_path}")
            # Usa custom_objects para resolver problemas de compatibilidade
            model = tf.keras.models.load_model(keras_path, compile=False)
            # Recompila o modelo para garantir compatibilidade
            model.compile(optimizer='adam', loss='mean_squared_error')
            print(f"✅ Modelo .keras carregado com sucesso!")
            return model
        except Exception as e:
            print(f"⚠️ Erro ao carregar modelo .keras: {e}")
            print("⚠️ Tentando carregar pesos em modelo recriado...")
            
            # Tenta recriar o modelo e carregar apenas os pesos
            try:
                new_model = create_compatible_model()
                # Tenta carregar apenas os pesos se o modelo completo falhar
                weights_path = keras_path.replace('.keras', '_weights.h5')
                if os.path.exists(weights_path):
                    new_model.load_weights(weights_path)
                    print(f"✅ Pesos carregados em modelo recriado!")
                    return new_model
                else:
                    # Se não tem arquivo de pesos separado, tenta carregar do .h5
                    print("⚠️ Tentando formatos alternativos...")
            except Exception as e2:
                print(f"⚠️ Falha ao recriar modelo: {e2}")
    
    # Tentativa 2: Carregar modelo .h5
    if os.path.exists(h5_path):
        try:
            print(f"✅ Tentando carregar modelo H5: {h5_path}")
            model = tf.keras.models.load_model(h5_path, compile=False)
            model.compile(optimizer='adam', loss='mean_squared_error')
            print(f"✅ Modelo .h5 carregado com sucesso!")
            return model
        except Exception as e:
            print(f"⚠️ Erro ao carregar modelo .h5: {e}")
    
    # Tentativa 3: Carregar modelo .joblib (último recurso)
    if os.path.exists(joblib_path):
        try:
            print(f"⚠️ Tentando carregar modelo joblib: {joblib_path}")
            import joblib
            model = joblib.load(joblib_path)
            print(f"✅ Modelo .joblib carregado (compatibilidade limitada)")
            return model
        except Exception as e:
            print(f"❌ Erro ao carregar modelo .joblib: {e}")
    
    # Se chegou aqui, nenhum formato funcionou
    available_files = []
    base_dir = os.path.dirname(keras_path) if os.path.dirname(keras_path) else 'lstm_files'
    if os.path.exists(base_dir):
        available_files = [f for f in os.listdir(base_dir) if f.startswith('lstm_model')]
    
    raise FileNotFoundError(f"❌ Não foi possível carregar modelo em nenhum formato. "
                          f"Arquivos disponíveis: {available_files}. "
                          f"Considere retreinar o modelo.")


def create_sequences(data, seq_length):
    """Cria sequências para treinamento do LSTM"""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def train_bitcoin_lstm_internal():
    """Treina o modelo LSTM para predição de Bitcoin internamente"""
    try:
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        print("🚀 Iniciando treinamento do modelo LSTM para Bitcoin...")
        
        # 1. Baixar dados do Bitcoin
        print("📊 Baixando dados do Bitcoin...")
        end_date = datetime.date.today()

        # Tenta diferentes períodos de dados em ordem decrescente
        periods_to_try = [2000, 1000, 500, 365, 180, 90]
        data = None

        for days in periods_to_try:
            start_date = end_date - datetime.timedelta(days=days)
            print(f"Tentando baixar {days} dias de dados de {start_date} até {end_date}")
            data = fetch_bitcoin_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

            if data is not None and len(data) > 0:
                print(f"✅ Sucesso! Baixados {len(data)} registros com {days} dias")
                break
            else:
                print(f"⚠️ Falha com {days} dias - dados vazios ou erro na fonte")
        
        # Se ainda não conseguiu dados depois das tentativas
        if data is None or len(data) == 0:
            print("⚠️ Falha ao obter dados de fontes externas")
        
        # Se ainda não tem dados, gera dados sintéticos para demonstração
        if data is None or len(data) == 0:
            print("⚠️ Gerando dados sintéticos para demonstração...")
            
            import pandas as pd
            
            # Gera dados sintéticos baseados em padrões realistas do Bitcoin
            np.random.seed(42)  # Para reprodutibilidade
            days = 200
            dates = pd.date_range(start=datetime.date.today() - datetime.timedelta(days=days), 
                                end=datetime.date.today(), freq='D')
            
            # Simula movimento de preços do Bitcoin (baseado em padrões históricos)
            initial_price = 45000
            prices = [initial_price]
            
            for i in range(1, len(dates)):
                # Movimento aleatório com tendência ligeiramente positiva
                change = np.random.normal(0.001, 0.03)  # 0.1% média, 3% volatilidade
                new_price = prices[-1] * (1 + change)
                
                # Evita preços muito baixos ou muito altos
                new_price = max(20000, min(100000, new_price))
                prices.append(new_price)
            
            # Cria DataFrame compatível com yfinance
            data = pd.DataFrame({
                'Open': prices,
                'High': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
                'Low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
                'Close': prices,
                'Volume': [np.random.randint(10000, 50000) for _ in prices]
            }, index=dates)
            
            print(f"✅ Dados sintéticos gerados: {len(data)} registros")
            print("⚠️ ATENÇÃO: Usando dados sintéticos para demonstração. Para produção, resolva a conectividade com Yahoo Finance.")
        
        if len(data) < 30:
            raise Exception(f"Dados insuficientes para treinamento. Obtidos apenas {len(data)} registros. Mínimo necessário: 30 para treinamento básico.")
        
        print(f"✅ Dados finais: {len(data)} registros de {data.index[0].date()} até {data.index[-1].date()}")
        
        # 2. Preparar dados
        print("🔧 Preparando dados...")
        close_prices = data['Close'].values.reshape(-1, 1)
        
        # Normalizar dados
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)
        
        # Criar sequências - ajusta sequence_length baseado na quantidade de dados
        if len(data) >= 200:
            sequence_length = 40
        elif len(data) >= 100:
            sequence_length = 20
        else:
            sequence_length = 10
            
        print(f"Usando sequence_length: {sequence_length} para {len(data)} registros")
        
        X, y = create_sequences(scaled_data, sequence_length)
        
        if len(X) < 10:
            raise Exception(f"Sequências insuficientes para treinamento. Obtidas {len(X)} sequências. Mínimo necessário: 10.")
        
        # Dividir em treino e teste
        train_size = int(len(X) * 0.8)
        if train_size < 5:
            train_size = len(X) - 2  # Deixa pelo menos 2 para teste
            
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Reshape para LSTM [samples, time steps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        print(f"✅ Dados preparados - Treino: {X_train.shape}, Teste: {X_test.shape}")
        
        # 3. Construir modelo
        print("🏗️ Construindo modelo LSTM...")
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, 1),
                               kernel_initializer='glorot_uniform',  # Mais compatível
                               recurrent_initializer='orthogonal'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50, return_sequences=True,
                               kernel_initializer='glorot_uniform',
                               recurrent_initializer='orthogonal'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50,
                               kernel_initializer='glorot_uniform',
                               recurrent_initializer='orthogonal'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, kernel_initializer='glorot_uniform')
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        print("✅ Modelo construído")
        
        # 4. Treinar modelo
        print("🎯 Treinando modelo...")
        start_time = datetime.datetime.now()
        
        # Ajusta epochs baseado na quantidade de dados
        if len(data) >= 500:
            epochs = 20
        elif len(data) >= 200:
            epochs = 15
        else:
            epochs = 10
            
        print(f"Treinando com {epochs} epochs para {len(data)} registros de dados")
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=min(32, len(X_train) // 2),  # Ajusta batch_size também
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        end_time = datetime.datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        print(f"✅ Treinamento concluído em {training_duration:.2f} segundos")
        
        # 5. Avaliar modelo
        print("📈 Avaliando modelo...")
        train_pred = model.predict(X_train, verbose=0)
        test_pred = model.predict(X_test, verbose=0)
        
        # Desnormalizar predições
        train_pred = scaler.inverse_transform(train_pred)
        test_pred = scaler.inverse_transform(test_pred)
        y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calcular métricas
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
        test_mae = mean_absolute_error(y_test_actual, test_pred)
        test_r2 = r2_score(y_test_actual, test_pred)
        
        print(f"📊 Métricas de Teste:")
        print(f"   RMSE: {test_rmse:.2f}")
        print(f"   MAE: {test_mae:.2f}")
        print(f"   R²: {test_r2:.4f}")
        
        # 6. Salvar modelo e scaler
        print("💾 Salvando modelo e scaler...")
        
        # Criar diretório se não existir
        os.makedirs('lstm_files', exist_ok=True)
        
        # Salvar modelo no formato Keras nativo (CORRETO)
        try:
            model.save('lstm_files/lstm_model.keras')
            print("✅ Modelo salvo em formato .keras")
        except Exception as e:
            print(f"⚠️ Erro ao salvar .keras: {e}")
        
        # Salvar também em H5 (formato mais compatível)
        try:
            model.save('lstm_files/lstm_model.h5')
            print("✅ Modelo salvo em formato .h5")
        except Exception as e:
            print(f"⚠️ Erro ao salvar .h5: {e}")
        
        # Salvar apenas os pesos (para fallback)
        try:
            model.save_weights('lstm_files/lstm_model_weights.h5')
            print("✅ Pesos do modelo salvos separadamente")
        except Exception as e:
            print(f"⚠️ Erro ao salvar pesos: {e}")
        
        # Também salva em joblib para compatibilidade (pode não funcionar, mas tenta)
        try:
            joblib.dump(model, 'lstm_files/lstm_model.joblib')
            print("✅ Modelo também salvo em formato joblib")
        except Exception as e:
            print(f"⚠️ Não foi possível salvar em joblib: {e}")
        
        # Salvar scaler
        joblib.dump(scaler, 'lstm_files/scaler.joblib')
        
        # Salvar configuração do modelo para recriação
        model_config = {
            'sequence_length': sequence_length,
            'architecture': 'LSTM-3layers-50units',
            'optimizer': 'adam',
            'loss': 'mean_squared_error',
            'layers': [
                {'type': 'LSTM', 'units': 50, 'return_sequences': True},
                {'type': 'Dropout', 'rate': 0.2},
                {'type': 'LSTM', 'units': 50, 'return_sequences': True},
                {'type': 'Dropout', 'rate': 0.2},
                {'type': 'LSTM', 'units': 50},
                {'type': 'Dropout', 'rate': 0.2},
                {'type': 'Dense', 'units': 1}
            ]
        }
        
        with open('lstm_files/model_config.json', 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Salvar métricas
        metrics = {
            'training_date': datetime.datetime.now().isoformat(),
            'rmse': float(test_rmse),
            'mae': float(test_mae),
            'r2': float(test_r2),
            'training_duration': training_duration,
            'data_points_used': len(data),
            'sequence_length': sequence_length,
            'train_size': train_size,
            'test_size': len(X_test)
        }
        
        with open('lstm_files/training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("✅ Modelo salvo em:")
        print("   - lstm_files/lstm_model.keras (FORMATO RECOMENDADO)")
        print("   - lstm_files/lstm_model.h5 (formato alternativo)")
        print("   - lstm_files/lstm_model_weights.h5 (apenas pesos)")
        print("   - lstm_files/scaler.joblib")
        print("   - lstm_files/model_config.json (configuração)")
        print("   - lstm_files/training_metrics.json")
        
        return {
            'success': True,
            'metrics': metrics,
            'message': f"Treinamento concluído com sucesso! RMSE: {test_rmse:.2f}, R²: {test_r2:.4f}"
        }
        
    except Exception as e:
        print(f"❌ Erro durante o treinamento: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


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


class TrainingResponse(BaseModel):
    status: str
    message: str
    training_started: str
    estimated_duration: str
    details: Dict


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
        ) else "needs_training",
        "available_endpoints": {
            "/predict": "Fazer predição do próximo dia do Bitcoin",
            "/model-check": "Verificar status detalhado dos arquivos do modelo",
            "/model-info": "Informações sobre o modelo",
            "/train": "Treinar um novo modelo LSTM (pode demorar ~5-10 minutos)",
            "/evaluate": "Avaliar métricas do modelo treinado",
            "/health": "Health check da aplicação",
            "/metrics": "Métricas Prometheus",
            "/docs": "Documentação Swagger da API"
        }
    }


# Rota para checagem detalhada dos arquivos do modelo
@app.get("/model-check")
async def model_check() -> Dict:
    """Verifica a existência dos arquivos do modelo e retorna logs detalhados"""
    logs = []
    # Prioridade: arquivo .keras
    model_path = 'lstm_files/lstm_model.keras'
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
        # Verifica se existe o modelo .keras (prioridade)
        if os.path.exists(model_path):
            result["model_exists"] = True
            file_size = os.path.getsize(model_path)
            logs.append(f"✅ Modelo .keras encontrado: {model_path} (tamanho: {file_size} bytes)")
        # Fallback para .h5
        elif os.path.exists('lstm_files/lstm_model.h5'):
            model_path = 'lstm_files/lstm_model.h5'
            result["model_path"] = os.path.abspath(model_path)
            result["model_exists"] = True
            file_size = os.path.getsize(model_path)
            logs.append(f"✅ Modelo .h5 encontrado: {model_path} (tamanho: {file_size} bytes)")
        # Último recurso: .joblib
        elif os.path.exists('lstm_files/lstm_model.joblib'):
            model_path = 'lstm_files/lstm_model.joblib'
            result["model_path"] = os.path.abspath(model_path)
            result["model_exists"] = True
            file_size = os.path.getsize(model_path)
            logs.append(f"⚠️ Apenas modelo .joblib encontrado: {model_path} (tamanho: {file_size} bytes)")
            logs.append("⚠️ Recomendado: Treinar novamente para gerar modelo .keras")
        else:
            logs.append(f"❌ Nenhum modelo encontrado (.keras, .h5, .joblib)")
            logs.append(f"❌ Caminho verificado: {os.path.abspath(model_path)}")

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
                logs.append(f"✅ load_model_safely OK para {model_path}")
                logs.append(f"Tipo do modelo: {type(model)}")
            except Exception as e:
                logs.append(f"❌ Erro ao carregar {model_path}: {str(e)}")
                
        if result["scaler_exists"]:
            try:
                scaler = joblib.load(scaler_path)
                logs.append(f"✅ joblib.load OK para {scaler_path}")
                logs.append(f"Tipo do scaler: {type(scaler)}")
            except Exception as e:
                logs.append(f"❌ Erro ao carregar {scaler_path}: {str(e)}")
                
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

        model = load_model_safely('lstm_files/lstm_model.keras')
        scaler = joblib.load('lstm_files/scaler.joblib')

        end_date = datetime.date.today().strftime("%Y-%m-%d")
        start_date = (datetime.date.today() - datetime.timedelta(days=100)).strftime("%Y-%m-%d")

        data = fetch_bitcoin_data(start_date, end_date, cache_hours=1)

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

@app.get("/train", response_model=TrainingResponse)
async def train_model():
    """Rota para treinar o modelo LSTM"""
    try:
        print("🚀 Iniciando treinamento via API...")
        
        # Executa o treinamento
        result = train_bitcoin_lstm_internal()
        
        if result['success']:
            return TrainingResponse(
                status="success",
                message=result['message'],
                training_started=datetime.datetime.now().isoformat(),
                estimated_duration="50 epochs completed",
                details=result['metrics']
            )
        else:
            ERROR_COUNT.labels(endpoint="/train").inc()
            return TrainingResponse(
                status="error",
                message=f"Falha no treinamento: {result['error']}",
                training_started=datetime.datetime.now().isoformat(),
                estimated_duration="N/A",
                details={"error": result['error'], "traceback": result.get('traceback', '')}
            )
            
    except Exception as e:
        ERROR_COUNT.labels(endpoint="/train").inc()
        tb = traceback.format_exc()
        return TrainingResponse(
            status="error",
            message=f"Erro crítico no treinamento: {str(e)}",
            training_started=datetime.datetime.now().isoformat(),
            estimated_duration="N/A",
            details={"error": str(e), "traceback": tb}
        )

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
