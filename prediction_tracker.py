"""
Prediction Tracker & Model Evaluator
Tracks prediction accuracy over time and provides insights for improvement.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf


class PredictionTracker:
    """
    Tracks predictions vs actual outcomes to measure model accuracy.
    """
    
    def __init__(self, history_file='predictions/prediction_history.json'):
        self.history_file = Path(history_file)
        self.history = self._load_history()
    
    def _load_history(self):
        """Load existing prediction history."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {'predictions': [], 'evaluations': []}
    
    def _save_history(self):
        """Save prediction history."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def log_prediction(self, commodity, ticker, prediction_date, 
                       current_price, forecasts):
        """
        Log a new prediction for later evaluation.
        
        Args:
            commodity: Name (e.g., "Crude Oil WTI")
            ticker: Yahoo Finance ticker (e.g., "CL=F")
            prediction_date: When prediction was made
            current_price: Price at prediction time
            forecasts: Dict with '1d', '5d', '10d' predictions
        """
        record = {
            'commodity': commodity,
            'ticker': ticker,
            'prediction_date': prediction_date,
            'current_price': current_price,
            'forecasts': forecasts,
            'actuals': {},  # Will be filled in later
            'evaluated': False
        }
        
        self.history['predictions'].append(record)
        self._save_history()
    
    def log_predictions_from_json(self, predictions_json_path):
        """Load predictions from the daily output JSON."""
        with open(predictions_json_path, 'r') as f:
            data = json.load(f)
        
        prediction_date = data['generated_at'][:10]  # YYYY-MM-DD
        
        for commodity, info in data['predictions'].items():
            self.log_prediction(
                commodity=commodity,
                ticker=info['ticker'],
                prediction_date=prediction_date,
                current_price=info['current_price'],
                forecasts=info['forecasts']
            )
        
        print(f"Logged {len(data['predictions'])} predictions from {prediction_date}")
    
    def fetch_actual_price(self, ticker, date):
        """Fetch actual price for a specific date."""
        try:
            target_date = pd.to_datetime(date)
            # Fetch a few days around target in case of weekends/holidays
            start = target_date - timedelta(days=5)
            end = target_date + timedelta(days=5)
            
            data = yf.download(ticker, start=start, end=end, progress=False)
            
            if data.empty:
                return None
            
            # Handle multi-index columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Find closest date
            data.index = pd.to_datetime(data.index)
            
            if target_date in data.index:
                return float(data.loc[target_date, 'Close'])
            
            # Find nearest date
            nearest_idx = data.index.get_indexer([target_date], method='nearest')[0]
            if nearest_idx >= 0 and nearest_idx < len(data):
                return float(data.iloc[nearest_idx]['Close'])
            
            return None
            
        except Exception as e:
            print(f"Error fetching {ticker} for {date}: {e}")
            return None
    
    def evaluate_predictions(self):
        """
        Evaluate past predictions by comparing to actual prices.
        """
        today = datetime.now().date()
        evaluated_count = 0
        
        for pred in self.history['predictions']:
            if pred['evaluated']:
                continue
            
            pred_date = datetime.strptime(pred['prediction_date'], '%Y-%m-%d').date()
            
            # Check each horizon
            for horizon_key in ['1d', '5d', '10d']:
                horizon_days = int(horizon_key[:-1])
                target_date = pred_date + timedelta(days=horizon_days)
                
                # Only evaluate if target date has passed
                if target_date > today:
                    continue
                
                # Skip if already have actual
                if horizon_key in pred['actuals']:
                    continue
                
                # Fetch actual price
                actual = self.fetch_actual_price(
                    pred['ticker'], 
                    target_date.strftime('%Y-%m-%d')
                )
                
                if actual is not None:
                    pred['actuals'][horizon_key] = {
                        'price': round(actual, 2),
                        'date': target_date.strftime('%Y-%m-%d')
                    }
                    evaluated_count += 1
            
            # Mark as fully evaluated if all horizons are done
            if all(h in pred['actuals'] for h in ['1d', '5d', '10d']):
                pred['evaluated'] = True
        
        self._save_history()
        print(f"Evaluated {evaluated_count} new data points")
        return evaluated_count
    
    def calculate_metrics(self):
        """
        Calculate accuracy metrics for all predictions.
        """
        results = {}
        
        for pred in self.history['predictions']:
            commodity = pred['commodity']
            
            if commodity not in results:
                results[commodity] = {
                    '1d': {'errors': [], 'directions': []},
                    '5d': {'errors': [], 'directions': []},
                    '10d': {'errors': [], 'directions': []}
                }
            
            for horizon in ['1d', '5d', '10d']:
                if horizon not in pred['actuals']:
                    continue
                
                predicted = pred['forecasts'][horizon]['price']
                actual = pred['actuals'][horizon]['price']
                current = pred['current_price']
                
                # Percentage error
                pct_error = abs(predicted - actual) / actual * 100
                results[commodity][horizon]['errors'].append(pct_error)
                
                # Directional accuracy
                pred_direction = 1 if predicted > current else -1
                actual_direction = 1 if actual > current else -1
                correct = 1 if pred_direction == actual_direction else 0
                results[commodity][horizon]['directions'].append(correct)
        
        # Calculate summary metrics
        summary = {}
        for commodity, horizons in results.items():
            summary[commodity] = {}
            for horizon, data in horizons.items():
                if len(data['errors']) > 0:
                    summary[commodity][horizon] = {
                        'mape': round(np.mean(data['errors']), 2),
                        'directional_accuracy': round(np.mean(data['directions']) * 100, 1),
                        'sample_size': len(data['errors'])
                    }
        
        return summary
    
    def get_performance_report(self):
        """
        Generate a human-readable performance report.
        """
        metrics = self.calculate_metrics()
        
        report = []
        report.append("=" * 60)
        report.append("MODEL PERFORMANCE REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("=" * 60)
        
        for commodity, horizons in metrics.items():
            report.append(f"\n{commodity}")
            report.append("-" * 40)
            
            for horizon, data in horizons.items():
                report.append(
                    f"  {horizon}: MAPE={data['mape']:.1f}%, "
                    f"Direction={data['directional_accuracy']:.0f}%, "
                    f"n={data['sample_size']}"
                )
        
        report.append("\n" + "=" * 60)
        report.append("INTERPRETATION:")
        report.append("  MAPE < 5%: Excellent")
        report.append("  MAPE 5-10%: Good")
        report.append("  MAPE > 10%: Needs improvement")
        report.append("  Direction > 55%: Better than random")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def get_worst_predictions(self, n=5):
        """Find the worst predictions to understand failure modes."""
        errors = []
        
        for pred in self.history['predictions']:
            for horizon in ['1d', '5d', '10d']:
                if horizon in pred['actuals']:
                    predicted = pred['forecasts'][horizon]['price']
                    actual = pred['actuals'][horizon]['price']
                    pct_error = abs(predicted - actual) / actual * 100
                    
                    errors.append({
                        'commodity': pred['commodity'],
                        'date': pred['prediction_date'],
                        'horizon': horizon,
                        'predicted': predicted,
                        'actual': actual,
                        'error_pct': round(pct_error, 2)
                    })
        
        # Sort by error
        errors.sort(key=lambda x: x['error_pct'], reverse=True)
        return errors[:n]
    
    def export_for_analysis(self, output_path='predictions/analysis_data.csv'):
        """Export prediction data for analysis in Excel/Python."""
        rows = []
        
        for pred in self.history['predictions']:
            for horizon in ['1d', '5d', '10d']:
                row = {
                    'commodity': pred['commodity'],
                    'prediction_date': pred['prediction_date'],
                    'horizon': horizon,
                    'price_at_prediction': pred['current_price'],
                    'predicted_price': pred['forecasts'][horizon]['price'],
                    'predicted_change_pct': pred['forecasts'][horizon]['change_pct'],
                }
                
                if horizon in pred['actuals']:
                    actual = pred['actuals'][horizon]['price']
                    row['actual_price'] = actual
                    row['actual_change_pct'] = (actual - pred['current_price']) / pred['current_price'] * 100
                    row['error_pct'] = abs(row['predicted_price'] - actual) / actual * 100
                    row['direction_correct'] = (
                        (row['predicted_change_pct'] > 0) == (row['actual_change_pct'] > 0)
                    )
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} records to {output_path}")
        return df


def generate_performance_json(tracker, output_path='predictions/performance.json'):
    """
    Generate performance JSON for dashboard display.
    """
    metrics = tracker.calculate_metrics()
    worst = tracker.get_worst_predictions(10)
    
    output = {
        'generated_at': datetime.now().isoformat(),
        'metrics': metrics,
        'worst_predictions': worst,
        'total_predictions': len(tracker.history['predictions']),
        'evaluated_predictions': sum(
            1 for p in tracker.history['predictions'] if p['evaluated']
        )
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Performance data saved to {output_path}")
    return output


# =============================================================================
# MAIN - Run this daily AFTER the main predictions
# =============================================================================

if __name__ == "__main__":
    tracker = PredictionTracker()
    
    # 1. Log today's predictions
    predictions_file = Path('predictions/latest_predictions.json')
    if predictions_file.exists():
        tracker.log_predictions_from_json(predictions_file)
    
    # 2. Evaluate past predictions (fetch actuals)
    print("\nEvaluating past predictions...")
    tracker.evaluate_predictions()
    
    # 3. Generate report
    print("\n" + tracker.get_performance_report())
    
    # 4. Export data
    generate_performance_json(tracker)
    tracker.export_for_analysis()
    
    # 5. Show worst predictions
    print("\nWORST PREDICTIONS (to investigate):")
    for w in tracker.get_worst_predictions(5):
        print(f"  {w['commodity']} {w['date']} {w['horizon']}: "
              f"predicted ${w['predicted']}, actual ${w['actual']} "
              f"(error: {w['error_pct']:.1f}%)")
