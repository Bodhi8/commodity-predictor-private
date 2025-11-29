# US Commodity Price Prediction Model

A deep learning framework for predicting US commodity prices using LSTM, BiLSTM, Transformer, and hybrid architectures.

## Features

- **Multiple Data Sources**: Yahoo Finance futures, World Bank Pink Sheet, FRED
- **Comprehensive Feature Engineering**: 
  - Lag features
  - Moving averages (SMA, EMA)
  - Volatility measures
  - Momentum indicators (RSI, MACD, Bollinger Bands)
  - Cross-commodity relationships
  - Calendar features
- **State-of-the-Art Models**:
  - LSTM (Long Short-Term Memory)
  - BiLSTM (Bidirectional LSTM)
  - CNN-LSTM (Convolutional + LSTM hybrid)
  - Transformer (Self-attention based)
  - LSTM-Transformer Hybrid

## Supported Commodities

### Energy
- Crude Oil WTI, Brent Crude
- Natural Gas
- Gasoline, Heating Oil

### Precious Metals
- Gold, Silver
- Platinum, Palladium

### Base Metals
- Copper

### Agriculture
- Corn, Wheat, Soybeans
- Coffee, Sugar, Cotton, Cocoa

### Livestock
- Live Cattle, Lean Hogs

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from train import run_experiment

# Run full experiment
results, predictions = run_experiment(
    target_commodity='Crude Oil WTI',
    start_date='2015-01-01',
    sequence_length=60,       # 60-day lookback
    prediction_horizon=1,     # 1-day ahead prediction
    models_to_train=['lstm', 'bilstm', 'transformer', 'lstm_transformer'],
    epochs=100,
    batch_size=32
)
```

## Project Structure

```
commodity_predictor/
├── data_loader.py          # Data downloading from multiple sources
├── feature_engineering.py  # Technical indicators & feature creation
├── models.py               # Deep learning model architectures
├── train.py                # Training & evaluation pipeline
├── requirements.txt        # Dependencies
└── results/                # Output directory for plots & metrics
```

## Usage Examples

### 1. Load Data Only

```python
from data_loader import CommodityDataLoader

loader = CommodityDataLoader(start_date='2020-01-01')
data = loader.load_all_data(source='yahoo')
loader.get_data_summary()

# Create composite index
index = loader.create_composite_index(data, method='equal')
```

### 2. Custom Feature Engineering

```python
from feature_engineering import FeatureEngineer

fe = FeatureEngineer()
features = fe.build_features(
    data,
    include_lags=True,
    include_returns=True,
    include_ma=True,
    include_volatility=True,
    include_momentum=True,
    include_rsi=True,
    include_bb=True,
    include_macd=True
)
```

### 3. Train Single Model

```python
from models import get_model, ModelTrainer
import torch

# Prepare data (see feature_engineering.py)
X_train, y_train, X_val, y_val, X_test, y_test = fe.prepare_sequences(
    features, target_col='Gold', sequence_length=60
)

# Create and train model
model = get_model('bilstm', input_size=X_train.shape[2])
trainer = ModelTrainer(model, device='cpu')
trainer.train(X_train, y_train, X_val, y_val, epochs=50)

# Predict
predictions = trainer.predict(X_test)
```

### 4. Predict Different Commodities

```python
# Energy
run_experiment(target_commodity='Natural Gas', ...)

# Precious Metals
run_experiment(target_commodity='Gold', ...)

# Agriculture
run_experiment(target_commodity='Corn', ...)
```

## Model Hyperparameters

| Model | Hidden Size | Layers | Dropout | Learning Rate |
|-------|-------------|--------|---------|---------------|
| LSTM | 64 | 2 | 0.2 | 0.001 |
| BiLSTM | 64 | 2 | 0.2 | 0.001 |
| Transformer | d_model=64 | 2 | 0.1 | 0.001 |
| Hybrid | 64 | 2+2 | 0.2 | 0.001 |

## Evaluation Metrics

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of Determination
- **Directional Accuracy**: % of correct up/down predictions

## Tips for Best Results

1. **Sequence Length**: 30-90 days works well for most commodities
2. **Feature Selection**: More features isn't always better; monitor for overfitting
3. **Data Quality**: Fill missing values carefully; commodities have market holidays
4. **Ensemble**: Consider averaging predictions from multiple models
5. **External Data**: Add macroeconomic indicators for better performance

## References

- [LSTM for Commodity Price Prediction](https://jfin-swufe.springeropen.com/articles/10.1186/s40854-024-00637-z)
- [Transformer vs LSTM Comparison](https://www.scitepress.org/Papers/2024/129811/129811.pdf)
- [World Bank Commodity Data](https://www.worldbank.org/en/research/commodity-markets)

## License

MIT License
