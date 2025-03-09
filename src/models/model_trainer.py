import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

class ModelTrainer:
    def __init__(self, sequence_length, n_features, model_dir="models", logs_dir="logs"):
        """
        Initialize model trainer
        
        Parameters:
        - sequence_length: Length of input sequences
        - n_features: Number of features
        - model_dir: Directory to save models
        - logs_dir: Directory to save training logs
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model_dir = model_dir
        self.logs_dir = logs_dir
        self.models = {}
        self.ensemble = None
        self.history = {
            'train_scores': [],
            'val_scores': [],
            'best_params': [],
            'feature_importance': [],
            'classification_metrics': []  # Added to track classification metrics
        }
        
        # Create directories if they don't exist
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create metrics tracking file if it doesn't exist
        self.metrics_file = os.path.join(model_dir, "training_metrics.csv")
        if not os.path.exists(self.metrics_file):
            metrics_df = pd.DataFrame(columns=[
                'timestamp', 'model_type', 'epochs', 'best_epoch',
                'train_mse', 'val_mse', 'test_mse',
                'train_r2', 'val_r2', 'test_r2',
                'accuracy', 'precision', 'recall', 'f1_score',
                'training_time', 'notes'
            ])
            metrics_df.to_csv(self.metrics_file, index=False)
    
    def create_base_models(self):
        """Create base models for ensemble"""
        models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gbr': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0, random_state=42)
        }
        return models
    
    def reshape_data(self, X):
        """Reshape 3D data to 2D for scikit-learn models"""
        if len(X.shape) == 3:  # (samples, sequence_length, features)
            return X.reshape(X.shape[0], -1)
        return X
        
    def calculate_classification_metrics(self, y_true, y_pred, y_prev=None):
        """
        Calculate classification metrics based on price movement direction:
        - If price goes up: class 1
        - If price goes down or stays the same: class 0
        
        Parameters:
        - y_true: True target values
        - y_pred: Predicted target values
        - y_prev: Previous price values (if None, will be derived from y_true)
        
        Returns:
        - Dictionary with classification metrics
        """
        # Use previous values if provided, otherwise derive from y_true
        if y_prev is None:
            # For validation/test data, we need to know the previous values
            # This is approximate - ideally we'd have the actual previous values
            y_prev = np.roll(y_true, 1)
            y_prev[0] = y_true[0]  # Avoid using rolled value for first element
        
        # Calculate true price movement directions
        y_true_dir = (y_true > y_prev).astype(int)
        
        # Calculate predicted price movement directions
        y_pred_dir = (y_pred > y_prev).astype(int)
        
        # Calculate classification metrics
        metrics = {
            'accuracy': accuracy_score(y_true_dir, y_pred_dir),
            'precision': precision_score(y_true_dir, y_pred_dir, zero_division=0),
            'recall': recall_score(y_true_dir, y_pred_dir, zero_division=0),
            'f1': f1_score(y_true_dir, y_pred_dir, zero_division=0)
        }
        
        return metrics
    
    def train_epoch(self, X_train, y_train, X_val, y_val, epoch, param_grid=None):
        """Train one epoch with hyperparameter tuning and model selection"""
        print(f"\nTraining Epoch {epoch+1}")
        
        # Reshape data if needed
        X_train_2d = self.reshape_data(X_train)
        X_val_2d = self.reshape_data(X_val)
        
        # Create models dictionary if not already created
        if not self.models:
            self.models = self.create_base_models()
        
        # Time series split for cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Default parameter grid if none provided
        if param_grid is None:
            param_grid = {
                'rf': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                },
                'gbr': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7]
                },
                'ridge': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            }
            
        # Train and tune each model
        epoch_results = {}
        best_params = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Grid search with time series CV
            grid = GridSearchCV(
                estimator=model,
                param_grid=param_grid[name],
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            # Fit grid search
            grid.fit(X_train_2d, y_train)
            
            # Get best model
            best_model = grid.best_estimator_
            self.models[name] = best_model
            
            # Store best parameters
            best_params[name] = grid.best_params_
            
            # Evaluate on validation set
            val_pred = best_model.predict(X_val_2d)
            val_mse = mean_squared_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            print(f"  Best params: {grid.best_params_}")
            print(f"  Validation MSE: {val_mse:.6f}")
            print(f"  Validation R²: {val_r2:.6f}")
            
            # Store scores
            epoch_results[name] = {
                'mse': val_mse,
                'r2': val_r2,
                'model': best_model
            }
            
            # Store feature importance if available
            if hasattr(best_model, 'feature_importances_'):
                epoch_results[name]['feature_importances'] = best_model.feature_importances_
        
        # Create ensemble from best models
        best_models = [(name, model['model']) for name, model in epoch_results.items()]
        self.ensemble = VotingRegressor(estimators=best_models)
        
        # Train ensemble
        print("Training ensemble model...")
        self.ensemble.fit(X_train_2d, y_train)
        
        # Evaluate ensemble
        ensemble_val_pred = self.ensemble.predict(X_val_2d)
        ensemble_val_mse = mean_squared_error(y_val, ensemble_val_pred)
        ensemble_val_r2 = r2_score(y_val, ensemble_val_pred)
        
        print(f"Ensemble Validation MSE: {ensemble_val_mse:.6f}")
        print(f"Ensemble Validation R²: {ensemble_val_r2:.6f}")
        
        # Calculate classification metrics for validation data
        # Get previous values from y_train (last elements before validation set)
        val_prev_values = np.roll(y_val, 1)
        val_prev_values[0] = y_train[-1]  # First element should use last training value
        
        class_metrics = self.calculate_classification_metrics(y_val, ensemble_val_pred, val_prev_values)
        print(f"Direction Prediction - Accuracy: {class_metrics['accuracy']:.4f}, "
              f"F1: {class_metrics['f1']:.4f}, "
              f"Precision: {class_metrics['precision']:.4f}, "
              f"Recall: {class_metrics['recall']:.4f}")
        
        # Update history to include classification metrics
        self.history['classification_metrics'].append(class_metrics)
        
        # Update history
        self.history['train_scores'].append({name: model.score(X_train_2d, y_train) for name, model in self.models.items()})
        self.history['val_scores'].append({
            **{name: result['r2'] for name, result in epoch_results.items()}, 
            'ensemble': ensemble_val_r2
        })
        self.history['best_params'].append(best_params)
        
        # Save models for this epoch
        self.save_models(epoch)
        
        return {
            'ensemble_mse': ensemble_val_mse,
            'ensemble_r2': ensemble_val_r2,
            'classification_metrics': class_metrics,
            'models': epoch_results
        }
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, param_grid=None, notes=""):
        """Train models for multiple epochs"""
        results = []
        best_score = float('-inf')
        best_epoch = 0
        start_time = datetime.now()
        
        for epoch in range(epochs):
            # Skip parameter tuning every 5 epochs to save time
            current_param_grid = param_grid if (epoch % 5 == 0 or epoch == 0) else None
            
            # Train one epoch
            epoch_result = self.train_epoch(X_train, y_train, X_val, y_val, epoch, current_param_grid)
            results.append(epoch_result)
            
            # Check if this is the best epoch
            if epoch_result['ensemble_r2'] > best_score:
                best_score = epoch_result['ensemble_r2']
                best_epoch = epoch
                self.save_models(epoch, is_best=True)
            
            # Every 10 epochs, save learning curves
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                self.plot_learning_curves()
        
        # After training is complete:
        end_time = datetime.now()
        training_time = end_time - start_time
        
        # Get best epoch's results
        best_result = results[best_epoch]
        
        # Save metrics to CSV
        self.save_metrics_to_csv(
            epochs=epochs,
            best_epoch=best_epoch,
            train_scores=self.history['train_scores'][best_epoch],
            val_mse=best_result['ensemble_mse'],
            val_r2=best_result['ensemble_r2'],
            class_metrics=best_result['classification_metrics'],
            training_time=training_time,
            notes=notes
        )
        
        print(f"\nTraining complete. Best epoch: {best_epoch+1} with R²: {best_score:.6f}")
        
        # Load best model
        self.load_models(best_epoch, is_best=True)
        
        return {
            'results': results,
            'best_epoch': best_epoch,
            'best_score': best_score,
            'training_time': training_time,
            'classification_metrics': best_result['classification_metrics']
        }
    
    def predict(self, X):
        """Make predictions using the ensemble model"""
        if self.ensemble is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Reshape data if needed
        X_2d = self.reshape_data(X)
        
        # Make predictions
        predictions = self.ensemble.predict(X_2d)
        
        # Reshape to match expected output format
        return predictions.reshape(-1, 1)
    
    def save_models(self, epoch, is_best=False):
        """Save models to disk"""
        # Create epoch directory
        if is_best:
            epoch_dir = os.path.join(self.model_dir, "best")
        else:
            epoch_dir = os.path.join(self.model_dir, f"epoch_{epoch+1}")
        
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Save individual models
        for name, model in self.models.items():
            model_path = os.path.join(epoch_dir, f"{name}_model.joblib")
            joblib.dump(model, model_path)
        
        # Save ensemble model
        ensemble_path = os.path.join(epoch_dir, "ensemble_model.joblib")
        joblib.dump(self.ensemble, ensemble_path)
        
        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'epoch': epoch,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models': list(self.models.keys())
        }
        
        metadata_path = os.path.join(epoch_dir, "metadata.joblib")
        joblib.dump(metadata, metadata_path)
        
        # Save history
        history_path = os.path.join(self.logs_dir, "training_history.joblib")
        joblib.dump(self.history, history_path)
        
        if is_best:
            print(f"Saved best models to {epoch_dir}")
        else:
            print(f"Saved epoch {epoch+1} models to {epoch_dir}")
    
    def load_models(self, epoch=None, is_best=False):
        """Load models from disk"""
        if is_best:
            epoch_dir = os.path.join(self.model_dir, "best")
        else:
            if epoch is None:
                raise ValueError("Must provide epoch number or set is_best=True")
            epoch_dir = os.path.join(self.model_dir, f"epoch_{epoch+1}")
        
        # Load metadata
        metadata_path = os.path.join(epoch_dir, "metadata.joblib")
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.sequence_length = metadata['sequence_length']
            self.n_features = metadata['n_features']
        
        # Load individual models
        self.models = {}
        model_files = [f for f in os.listdir(epoch_dir) if f.endswith('_model.joblib') and not f.startswith('ensemble')]
        
        for model_file in model_files:
            name = model_file.split('_')[0]
            model_path = os.path.join(epoch_dir, model_file)
            self.models[name] = joblib.load(model_path)
        
        # Load ensemble model
        ensemble_path = os.path.join(epoch_dir, "ensemble_model.joblib")
        if os.path.exists(ensemble_path):
            self.ensemble = joblib.load(ensemble_path)
        
        # Load history
        history_path = os.path.join(self.logs_dir, "training_history.joblib")
        if os.path.exists(history_path):
            self.history = joblib.load(history_path)
        
        if is_best:
            print(f"Loaded best models from {epoch_dir}")
        else:
            print(f"Loaded epoch {epoch+1} models from {epoch_dir}")
    
    def save_metrics_to_csv(self, epochs, best_epoch, train_scores, val_mse, val_r2, 
                           class_metrics, training_time, notes="", test_mse=None, test_r2=None):
        """Save performance metrics to CSV file"""
        # Create new metrics row
        new_metrics = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'ensemble',
            'epochs': epochs,
            'best_epoch': best_epoch + 1,  # Add 1 for human-readable epoch number
            'train_mse': -1,  # Will be filled if available
            'val_mse': val_mse,
            'test_mse': test_mse if test_mse is not None else -1,
            'train_r2': train_scores.get('ensemble', -1),
            'val_r2': val_r2,
            'test_r2': test_r2 if test_r2 is not None else -1,
            'accuracy': class_metrics['accuracy'],
            'precision': class_metrics['precision'],
            'recall': class_metrics['recall'],
            'f1_score': class_metrics['f1'],
            'training_time': str(training_time),
            'notes': notes
        }
        
        # Read existing metrics, append new row, and save
        try:
            metrics_df = pd.read_csv(self.metrics_file)
            metrics_df = pd.concat([metrics_df, pd.DataFrame([new_metrics])], ignore_index=True)
        except:
            # If reading fails, create new DataFrame
            metrics_df = pd.DataFrame([new_metrics])
            
        # Save updated metrics
        metrics_df.to_csv(self.metrics_file, index=False)
        print(f"Performance metrics saved to {self.metrics_file}")
    
    def update_test_metrics(self, y_test, test_predictions):
        """Update metrics file with test set results"""
        # Calculate regression metrics
        test_mse = mean_squared_error(y_test, test_predictions)
        test_r2 = r2_score(y_test, test_predictions)
        
        # Calculate classification metrics
        test_prev_values = np.roll(y_test, 1)
        test_prev_values[0] = y_test[0]  # Avoid using rolled value for first element
        class_metrics = self.calculate_classification_metrics(
            y_test, test_predictions.ravel(), test_prev_values
        )
        
        # Read existing metrics
        try:
            metrics_df = pd.read_csv(self.metrics_file)
            if not metrics_df.empty:
                # Update the latest row with test metrics
                metrics_df.iloc[-1, metrics_df.columns.get_loc('test_mse')] = test_mse
                metrics_df.iloc[-1, metrics_df.columns.get_loc('test_r2')] = test_r2
                
                # Update classification metrics if they've improved
                if class_metrics['accuracy'] > metrics_df.iloc[-1]['accuracy']:
                    metrics_df.iloc[-1, metrics_df.columns.get_loc('accuracy')] = class_metrics['accuracy']
                    metrics_df.iloc[-1, metrics_df.columns.get_loc('precision')] = class_metrics['precision']
                    metrics_df.iloc[-1, metrics_df.columns.get_loc('recall')] = class_metrics['recall']
                    metrics_df.iloc[-1, metrics_df.columns.get_loc('f1_score')] = class_metrics['f1']
                
                # Save updated metrics
                metrics_df.to_csv(self.metrics_file, index=False)
                print(f"Test metrics updated in {self.metrics_file}")
        except Exception as e:
            print(f"Error updating test metrics: {e}")
            
        return {
            'test_mse': test_mse,
            'test_r2': test_r2,
            'classification_metrics': class_metrics
        }
    
    def plot_learning_curves(self):
        """Plot learning curves from training history"""
        if not self.history['val_scores']:
            print("No training history available")
            return
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Extract epochs and metrics
        epochs = list(range(1, len(self.history['val_scores']) + 1))
        
        # Plot regression metrics (R²)
        for model_name in self.history['val_scores'][0].keys():
            val_scores = [epoch_scores[model_name] for epoch_scores in self.history['val_scores']]
            ax1.plot(epochs, val_scores, '-o', label=f"{model_name.capitalize()} (R²)")
        
        # Add labels and legend for regression plot
        ax1.set_title("Regression Performance (R² Score)", fontsize=16)
        ax1.set_ylabel("R² Score", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot classification metrics
        if self.history['classification_metrics']:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            colors = ['blue', 'green', 'red', 'purple']
            
            for i, metric_name in enumerate(metrics):
                metric_values = [epoch_metrics[metric_name] for epoch_metrics in self.history['classification_metrics']]
                ax2.plot(epochs, metric_values, '-o', color=colors[i], label=metric_name.capitalize())
        
        # Add labels and legend for classification plot
        ax2.set_title("Classification Performance (Direction Prediction)", fontsize=16)
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Score", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Adjust layout and save
        plt.tight_layout()
        plot_path = os.path.join(self.logs_dir, "learning_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Learning curves saved to {plot_path}")
        plt.close()
        
    def export_stock_prediction_model(self, output_path):
        """
        Export a model compatible with the StockPredictionModel interface
        
        This creates a model file that can be loaded by the existing system.
        """
        from src.models.sklearn_model import StockPredictionModel
        
        # Create a StockPredictionModel instance
        model = StockPredictionModel(self.sequence_length, self.n_features, model_type='ensemble')
        
        # Set the model to our trained ensemble
        model.model = self.ensemble
        
        # Save the model
        model.save(output_path)
        
        print(f"Exported stock prediction model to {output_path}")
        
        return model
