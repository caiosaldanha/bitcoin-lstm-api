#!/usr/bin/env python3
"""
Script para treinar e salvar o modelo LSTM do Bitcoin corretamente
"""

import os
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import json
import datetime

def create_sequences(data, seq_length):
    """Cria sequências para treinamento do LSTM"""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def train_bitcoin_lstm():
    """Treina o modelo LSTM para predição de Bitcoin"""
    print("🚀 Iniciando treinamento do modelo LSTM para Bitcoin...")
    
    # 1. Baixar dados do Bitcoin
    print("📊 Baixando dados do Bitcoin...")
    ticker_symbol = "BTC-USD"
    end_date = datetime.date.today()
    
    # Tenta diferentes períodos de dados em ordem decrescente
    periods_to_try = [2000, 1000, 500, 365, 180, 90]
    data = None
    
    for days in periods_to_try:
        try:
            start_date = end_date - datetime.timedelta(days=days)
            print(f"Tentando baixar {days} dias de dados de {start_date} até {end_date}")
            
            # Tenta com diferentes configurações do yfinance
            data = yf.download(
                ticker_symbol, 
                start=start_date.strftime("%Y-%m-%d"), 
                end=end_date.strftime("%Y-%m-%d"),
                progress=False,
                show_errors=False,
                threads=True
            )
            
            if data is not None and len(data) > 0:
                print(f"✅ Sucesso! Baixados {len(data)} registros com {days} dias")
                break
            else:
                print(f"⚠️ Falha com {days} dias - dados vazios")
                
        except Exception as e:
            print(f"⚠️ Erro ao baixar {days} dias: {str(e)}")
            continue
    
    # Se ainda não conseguiu dados, tenta métodos alternativos
    if data is None or len(data) == 0:
        print("⚠️ Falha com yfinance, tentando métodos alternativos...")
        
        # Método alternativo 1: Forçar download com período menor
        try:
            end_date_str = datetime.date.today().strftime("%Y-%m-%d")
            start_date_str = (datetime.date.today() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
            
            ticker = yf.Ticker("BTC-USD")
            data = ticker.history(start=start_date_str, end=end_date_str)
            
            if data is not None and len(data) > 0:
                print(f"✅ Método alternativo funcionou! {len(data)} registros")
            else:
                print("❌ Método alternativo também falhou")
                
        except Exception as e:
            print(f"❌ Erro no método alternativo: {e}")
    
    # Se ainda não tem dados, gera dados sintéticos para demonstração
    if data is None or len(data) == 0:
        print("⚠️ Gerando dados sintéticos para demonstração...")
        
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
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 1),
             kernel_initializer='glorot_uniform',
             recurrent_initializer='orthogonal'),
        Dropout(0.2),
        LSTM(50, return_sequences=True,
             kernel_initializer='glorot_uniform',
             recurrent_initializer='orthogonal'),
        Dropout(0.2),
        LSTM(50,
             kernel_initializer='glorot_uniform',
             recurrent_initializer='orthogonal'),
        Dropout(0.2),
        Dense(1, kernel_initializer='glorot_uniform')
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("✅ Modelo construído")
    
    # 4. Treinar modelo
    print("🎯 Treinando modelo...")
    start_time = datetime.datetime.now()
    
    # Ajusta epochs baseado na quantidade de dados
    if len(X_train) >= 100:
        epochs = 20
    elif len(X_train) >= 50:
        epochs = 15
    else:
        epochs = 10
        
    batch_size = min(32, len(X_train) // 4) if len(X_train) >= 16 else len(X_train)
    
    print(f"Usando {epochs} epochs e batch_size {batch_size} para {len(X_train)} amostras de treino")
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
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
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
    train_mae = mean_absolute_error(y_train_actual, train_pred)
    test_mae = mean_absolute_error(y_test_actual, test_pred)
    train_r2 = r2_score(y_train_actual, train_pred)
    test_r2 = r2_score(y_test_actual, test_pred)
    
    print(f"📊 Métricas de Treinamento:")
    print(f"   RMSE: {train_rmse:.2f}")
    print(f"   MAE: {train_mae:.2f}")
    print(f"   R²: {train_r2:.4f}")
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
    
    return model, scaler, metrics

if __name__ == "__main__":
    try:
        model, scaler, metrics = train_bitcoin_lstm()
        print("\n🎉 Treinamento concluído com sucesso!")
        print(f"📈 RMSE final: {metrics['rmse']:.2f}")
        print(f"📈 R² final: {metrics['r2']:.4f}")
    except Exception as e:
        print(f"❌ Erro durante o treinamento: {e}")
        import traceback
        traceback.print_exc()
