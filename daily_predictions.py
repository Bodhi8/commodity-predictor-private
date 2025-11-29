"""
Daily Commodity Prediction Script
Fixed for latest yfinance data format.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pickle


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'commodities': {
        'CL=F': 'Crude Oil WTI',
        'GC=F': 'Gold',
        'SI=F': 'Silver',
        'NG=F': 'Natural Gas',
        'HG=F': 'Copper',
        'ZC=F': 'Corn',
        'ZW=F': 'Wheat',
    },
    'sequence_length': 60,
    'prediction_horizons': [1, 5, 10],
    'models_dir': 'trained_models',
    'output_dir': 'predictions',
    'data_lookback_days': 365,
}


# =============================================================================
# MODELS
# =============================================================================

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# =============================================================================
# DATA FUNCTIONS
# =============================================================================

def fetch_commodity_data(ticker, days=365):
    """Fetch latest commodity data from Yahoo Finance."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            print(f"  No data returned for {ticker}")
            return None
        
        # Handle multi-index columns (new yfinance format)
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten multi-index: ('Close', 'CL=F') -> just get Close
            data.columns = data.columns.get_level_values(0)
        
        # Get Close prices as a Series
        if 'Close' in data.columns:
            close_series = data['Close'].copy()
            # Make sure it's a Series with datetime index
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]
            return close_series
        else:
            print(f"  No 'Close' column found for {ticker}")
            return None
            
    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return None


def create_features(series):
    """Create features for a single commodity series."""
    # Ensure series is a proper pandas Series
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    
    df = pd.DataFrame({'price': series.values}, index=series.index)
    
    # Lag features
    for lag in [1, 2, 3, 5, 7, 14, 21]:
        df[f'lag_{lag}'] = df['price'].shift(lag)
    
    # Returns
    for period in [1, 5, 10, 21]:
        df[f'return_{period}d'] = df['price'].pct_change(period)
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        df[f'sma_{window}'] = df['price'].rolling(window).mean()
        df[f'price_to_sma_{window}'] = df['price'] / df[f'sma_{window}']
    
    # Volatility
    returns = df['price'].pct_change()
    for window in [5, 10, 21]:
        df[f'vol_{window}d'] = returns.rolling(window).std()
    
    # RSI
    delta = df['price'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Band position
    sma = df['price'].rolling(20).mean()
    std = df['price'].rolling(20).std()
    df['bb_position'] = (df['price'] - (sma - 2*std)) / (4*std)
    
    # MACD
    ema12 = df['price'].ewm(span=12).mean()
    ema26 = df['price'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    return df


def prepare_prediction_input(df, seq_length, scaler=None):
    """Prepare the most recent data for prediction."""
    df_clean = df.dropna()
    
    if len(df_clean) < seq_length:
        return None, None
    
    if scaler is None:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df_clean)
    else:
        scaled = scaler.transform(df_clean)
    
    # Get the last sequence
    X = scaled[-seq_length:].reshape(1, seq_length, -1)
    
    return X, scaler


# =============================================================================
# TRAINING & PREDICTION
# =============================================================================

def train_model(df, input_size, seq_length=60, epochs=30):
    """Train a BiLSTM model."""
    df_clean = df.dropna()
    
    if len(df_clean) < seq_length + 50:
        return None, None
    
    # Scale
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_clean)
    
    # Create sequences
    X, y = [], []
    for i in range(seq_length, len(scaled) - 1):
        X.append(scaled[i-seq_length:i])
        y.append(scaled[i+1, 0])  # Predict next day's price
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    # Train/val split
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    
    # Initialize model
    model = BiLSTMModel(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    return model, scaler


def load_or_train_model(commodity_name, features_df, model_dir, seq_length=60):
    """Load existing model or train a new one."""
    safe_name = commodity_name.replace(' ', '_').replace('/', '_')
    model_path = Path(model_dir) / f"{safe_name}_model.pt"
    scaler_path = Path(model_dir) / f"{safe_name}_scaler.pkl"
    
    input_size = len(features_df.columns)
    
    if model_path.exists() and scaler_path.exists():
        print(f"  Loading saved model")
        model = BiLSTMModel(input_size)
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    else:
        print(f"  Training new model...")
        model, scaler = train_model(features_df, input_size, seq_length)
        
        if model is None:
            return None, None
        
        # Save model
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    
    return model, scaler


def make_predictions(model, X, scaler, current_price, horizons=[1, 5, 10]):
    """Generate predictions for multiple horizons."""
    model.eval()
    predictions = {}
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        pred_scaled = model(X_tensor).numpy()[0, 0]
        
        # Inverse transform
        dummy = np.zeros((1, scaler.n_features_in_))
        dummy[0, 0] = pred_scaled
        pred_price = scaler.inverse_transform(dummy)[0, 0]
        
        # Calculate predicted change
        pct_change = (pred_price - current_price) / current_price
        
        for horizon in horizons:
            horizon_change = pct_change * np.sqrt(horizon)
            horizon_price = current_price * (1 + horizon_change)
            predictions[f'{horizon}d'] = {
                'price': round(float(horizon_price), 2),
                'change_pct': round(float(horizon_change * 100), 2)
            }
    
    return predictions


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_daily_predictions():
    """Main function to run daily predictions."""
    print("=" * 60)
    print(f"DAILY COMMODITY PREDICTIONS - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    results = {
        'generated_at': datetime.now().isoformat(),
        'predictions': {},
        'historical': {},
        'metadata': {
            'models': ['BiLSTM'],
            'horizons': CONFIG['prediction_horizons'],
            'sequence_length': CONFIG['sequence_length']
        }
    }
    
    successful = 0
    
    for ticker, name in CONFIG['commodities'].items():
        print(f"\nProcessing {name} ({ticker})...")
        
        # Fetch data
        price_series = fetch_commodity_data(ticker, CONFIG['data_lookback_days'])
        
        if price_series is None:
            print(f"  Skipping - no data")
            continue
            
        if len(price_series) < CONFIG['sequence_length'] + 50:
            print(f"  Skipping - only {len(price_series)} days of data")
            continue
        
        # Create features
        features_df = create_features(price_series)
        features_clean = features_df.dropna()
        
        if len(features_clean) < CONFIG['sequence_length']:
            print(f"  Skipping - insufficient clean data")
            continue
        
        # Get current price
        current_price = float(price_series.iloc[-1])
        
        # Load or train model
        model, scaler = load_or_train_model(
            name, features_clean, CONFIG['models_dir'], CONFIG['sequence_length']
        )
        
        if model is None:
            print(f"  Skipping - model training failed")
            continue
        
        # Prepare input and predict
        X, _ = prepare_prediction_input(features_clean, CONFIG['sequence_length'], scaler)
        
        if X is None:
            print(f"  Skipping - could not prepare input")
            continue
        
        # Make predictions
        predictions = make_predictions(
            model, X, scaler, current_price, CONFIG['prediction_horizons']
        )
        
        # Store results
        results['predictions'][name] = {
            'ticker': ticker,
            'current_price': round(current_price, 2),
            'last_updated': price_series.index[-1].strftime('%Y-%m-%d'),
            'forecasts': predictions
        }
        
        # Store recent historical data for charts
        recent_prices = price_series.tail(90)
        results['historical'][name] = {
            'dates': [d.strftime('%Y-%m-%d') for d in recent_prices.index],
            'prices': [round(float(p), 2) for p in recent_prices.values]
        }
        
        print(f"  Current: ${current_price:.2f}")
        for horizon, pred in predictions.items():
            direction = "↑" if pred['change_pct'] > 0 else "↓"
            print(f"  {horizon}: ${pred['price']:.2f} ({direction}{abs(pred['change_pct']):.1f}%)")
        
        successful += 1
    
    print(f"\n{'=' * 60}")
    print(f"Completed: {successful}/{len(CONFIG['commodities'])} commodities")
    print("=" * 60)
    
    # Save results
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'latest_predictions.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved predictions to {output_file}")
    
    # Save dated backup
    dated_file = output_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.json"
    with open(dated_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    results = run_daily_predictions()
