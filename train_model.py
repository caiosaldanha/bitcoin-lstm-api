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
    start_date = end_date - datetime.timedelta(days=2000)  # 2000 dias de dados
    
    data = yf.download(ticker_symbol, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    
    if len(data) < 100:
        raise Exception("Dados insuficientes para treinamento")
    
    print(f"✅ Dados baixados: {len(data)} registros de {start_date} até {end_date}")
    
    # 2. Preparar dados
    print("🔧 Preparando dados...")
    close_prices = data['Close'].values.reshape(-1, 1)
    
    # Normalizar dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    # Criar sequências
    sequence_length = 40
    X, y = create_sequences(scaled_data, sequence_length)
    
    # Dividir em treino e teste
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Reshape para LSTM [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    print(f"✅ Dados preparados - Treino: {X_train.shape}, Teste: {X_test.shape}")
    
    # 3. Construir modelo
    print("🏗️ Construindo modelo LSTM...")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("✅ Modelo construído")
    
    # 4. Treinar modelo
    print("🎯 Treinando modelo...")
    start_time = datetime.datetime.now()
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
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
    model.save('lstm_files/lstm_model.keras')
    model.save('lstm_files/lstm_model.h5')  # Formato alternativo
    
    # Também salva em joblib para compatibilidade (pode não funcionar, mas tenta)
    try:
        joblib.dump(model, 'lstm_files/lstm_model.joblib')
        print("✅ Modelo também salvo em formato joblib")
    except Exception as e:
        print(f"⚠️ Não foi possível salvar em joblib: {e}")
    
    # Salvar scaler
    joblib.dump(scaler, 'lstm_files/scaler.joblib')
    
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
    print("   - lstm_files/lstm_model.joblib (compatibilidade)")
    print("   - lstm_files/scaler.joblib")
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
