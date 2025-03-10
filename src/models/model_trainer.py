import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

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
        self.direction_classifier = None  # Added for direction prediction
        self.direction_classifier_version = 0  # Track version for changes
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
    
    def create_direction_features(self, X, y=None, y_prev=None):
        """
        Create specialized features for direction prediction
        These features focus on patterns relevant to price movement direction
        
        Parameters:
        - X: Input features (2D array)
        - y: Target values (optional)
        - y_prev: Previous price values (optional)
        
        Returns:
        - Enhanced feature array with direction-specific features
        """
        # Start with original features
        X_enhanced = X.copy()
        
        # Get number of samples and features
        n_samples = X.shape[0]
        
        # Calculate feature differences (momentum indicators)
        # For sequence data that has been flattened, we need to be careful with the feature positions
        n_features = self.n_features  # Original number of features per time step
        seq_length = self.sequence_length  # Number of time steps in a sequence
        
        # For flattened sequence data, the shape is (samples, sequence_length * n_features)
        # Each time step's features are grouped together
        
        # Add momentum-based features (difference between current and previous time steps)
        dir_features = []
        
        # For each sample
        for i in range(n_samples):
            sample_features = []
            
            # Extract the closing prices across the sequence
            # Assuming close price is the first feature in each time step
            close_prices = np.array([X[i, j*n_features] for j in range(seq_length)])
            
            # Calculate price differences between consecutive time steps
            price_diffs = np.diff(close_prices)
            
            # Calculate direction changes and streaks
            directions = np.sign(price_diffs)
            direction_changes = np.diff(directions)
            
            # Calculate other momentum indicators
            roc = (close_prices[-1] / close_prices[0] - 1) * 100  # Rate of change
            momentum = np.sum(directions)  # Sum of directions (positive = uptrend)
            volatility = np.std(price_diffs)  # Volatility
            
            # Add these as features
            sample_features.extend([
                roc,                             # Rate of change
                momentum,                        # Momentum score
                volatility,                      # Volatility
                np.count_nonzero(directions > 0),  # Number of up moves
                np.count_nonzero(directions < 0),  # Number of down moves
                np.max(np.where(directions > 0)[0]) if np.any(directions > 0) else 0,  # Position of last up move
                np.max(np.where(directions < 0)[0]) if np.any(directions < 0) else 0,  # Position of last down move
            ])
            
            # Add strength of last few moves
            sample_features.extend(price_diffs[-3:])  # Last 3 price differences
            
            dir_features.append(sample_features)
        
        # Convert to numpy array
        dir_features = np.array(dir_features)
        
        # Create enhanced feature set by combining original and new features
        X_with_dir_features = np.hstack([X, dir_features])
        
        return X_with_dir_features
            
    def _prepare_direction_data(self, X, y, y_prev=None):
        """
        Prepare data for training direction classifier
        
        Returns:
        - X_enhanced: Enhanced input features for direction prediction
        - y_direction: Binary labels (1=price up, 0=price down/same)
        """
        # Create previous price vector if not provided
        if y_prev is None:
            y_prev = np.roll(y, 1)
            y_prev[0] = y[0]
            
        # Create direction labels: 1 if price went up, 0 if price went down/stayed same
        y_direction = (y > y_prev).astype(int)
        
        # Create enhanced features specifically for direction prediction
        X_enhanced = self.create_direction_features(X, y, y_prev)
        
        return X_enhanced, y_direction

    def train_direction_classifier(self, X_train, y_train, X_val=None, y_val=None, iteration=0):
        """Train a dedicated model for price direction prediction"""
        print("\nTraining dedicated direction classifier...")
        
        # Prepare training data for direction prediction
        # Get previous values from the training set (shifted by 1)
        y_train_prev = np.roll(y_train, 1)
        y_train_prev[0] = y_train[0]  # First value uses itself as previous
        
        # Create binary labels and enhanced features
        X_train_dir, y_train_dir = self._prepare_direction_data(X_train, y_train, y_train_prev)
        
        # Prepare validation data if provided
        if X_val is not None and y_val is not None:
            y_val_prev = np.roll(y_val, 1)
            y_val_prev[0] = y_train[-1]  # First validation value uses last training value as previous
            X_val_dir, y_val_dir = self._prepare_direction_data(X_val, y_val, y_val_prev)
        else:
            X_val_dir, y_val_dir = None, None
        
        # Check class balance and print info
        class_counts = np.bincount(y_train_dir)
        print(f"Direction class distribution - Up: {class_counts[1]} ({class_counts[1]/len(y_train_dir):.2%}), " 
              f"Down/Same: {class_counts[0]} ({class_counts[0]/len(y_train_dir):.2%})")
        
        # Create classifier based on iteration to introduce variation
        if iteration % 3 == 0:
            print("Using RandomForest for direction classifier")
            # Use more trees and different parameters for later iterations
            n_estimators = 100 + (iteration // 3) * 50  # Increase estimators over time
            max_depth = None if iteration < 10 else (10 + iteration // 2)
            
            classifier = RandomForestClassifier(
                n_estimators=min(n_estimators, 300),  # Cap at 300 trees
                max_depth=max_depth,
                class_weight='balanced',
                random_state=42+iteration  # Use different random state for diversity
            )
        elif iteration % 3 == 1:
            print("Using GradientBoosting for direction classifier")
            classifier = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42+iteration
            )
        else:
            print("Using LogisticRegression for direction classifier")
            classifier = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                C=0.1 * (2 ** (iteration // 5)),  # Vary regularization over iterations
                random_state=42+iteration
            )
        
        # Apply data balancing for class imbalance
        if iteration % 2 == 0 and class_counts[0] / len(y_train_dir) > 0.65 or class_counts[1] / len(y_train_dir) > 0.65:
            print("Applying SMOTE to balance classes...")
            try:
                smote = SMOTE(random_state=42+iteration)
                X_train_dir_resampled, y_train_dir_resampled = smote.fit_resample(X_train_dir, y_train_dir)
                print(f"Data shape after SMOTE: {X_train_dir_resampled.shape}, y: {np.bincount(y_train_dir_resampled)}")
                
                # Train on resampled data
                classifier.fit(X_train_dir_resampled, y_train_dir_resampled)
            except Exception as e:
                print(f"SMOTE failed: {e}. Training on original data.")
                classifier.fit(X_train_dir, y_train_dir)
        else:
            # Train on original data
            classifier.fit(X_train_dir, y_train_dir)
        
        # Evaluate on training data
        train_pred_dir = classifier.predict(X_train_dir)
        train_acc = accuracy_score(y_train_dir, train_pred_dir)
        train_f1 = f1_score(y_train_dir, train_pred_dir)
        
        print(f"Direction classifier - Training accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
        
        # Evaluate on validation data if provided
        if X_val_dir is not None and y_val_dir is not None:
            val_pred_dir = classifier.predict(X_val_dir)
            val_acc = accuracy_score(y_val_dir, val_pred_dir)
            val_f1 = f1_score(y_val_dir, val_pred_dir)
            val_precision = precision_score(y_val_dir, val_pred_dir)
            val_recall = recall_score(y_val_dir, val_pred_dir)
            
            # Try to get probability predictions for AUC if supported
            try:
                val_proba = classifier.predict_proba(X_val_dir)[:, 1]
                val_auc = roc_auc_score(y_val_dir, val_proba)
                print(f"Direction classifier - Validation AUC: {val_auc:.4f}")
            except:
                val_auc = 0
                
            print(f"Direction classifier - Validation metrics: Accuracy={val_acc:.4f}, F1={val_f1:.4f}, "
                  f"Precision={val_precision:.4f}, Recall={val_recall:.4f}")
        
        # Store the trained classifier and increment version
        self.direction_classifier = classifier
        self.direction_classifier_version += 1
        
        return classifier
        
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
        
        # Train direction classifier with increasing frequency in later epochs
        # Early epochs: train every 5 epochs
        # Later epochs: train more frequently to refine the classifier
        train_classifier = False
        if epoch < 10:
            train_classifier = epoch % 5 == 0
        elif epoch < 30:
            train_classifier = epoch % 3 == 0
        else:
            train_classifier = epoch % 2 == 0
            
        if train_classifier:
            # Reshape data for sklearn if needed
            X_train_2d = self.reshape_data(X_train)
            X_val_2d = self.reshape_data(X_val)
            
            # Train direction classifier with current epoch number for variety
            self.train_direction_classifier(X_train_2d, y_train, X_val_2d, y_val, iteration=epoch)
        
        # Save models for this epoch
        self.save_models(epoch)
        
        return {
            'ensemble_mse': ensemble_val_mse,
            'ensemble_r2': ensemble_val_r2,
            'classification_metrics': class_metrics,
            'models': epoch_results,
            'direction_classifier_version': self.direction_classifier_version
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
        
        # Save direction classifier if it exists
        if self.direction_classifier is not None:
            dir_classifier_path = os.path.join(epoch_dir, "direction_classifier.joblib")
            joblib.dump(self.direction_classifier, dir_classifier_path)
        
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
        
        # Load direction classifier if it exists
        dir_classifier_path = os.path.join(epoch_dir, "direction_classifier.joblib")
        if os.path.exists(dir_classifier_path):
            self.direction_classifier = joblib.load(dir_classifier_path)
        
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
            'direction_classifier_version': self.direction_classifier_version,
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
        
        # Add the direction classifier and version
        model.direction_classifier = self.direction_classifier
        model.direction_classifier_version = self.direction_classifier_version
        
        # Embed the direction feature creation function in the model
        def create_direction_features(self, X):
            """Create direction-specific features for prediction"""
            # Start with original features
            X_copy = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X.copy()
            
            # Get number of samples
            n_samples = X_copy.shape[0]
            
            # Calculate feature differences (momentum indicators)
            n_features = self.n_features
            seq_length = self.sequence_length
            
            # Add momentum-based features (difference between current and previous time steps)
            dir_features = []
            
            # For each sample
            for i in range(n_samples):
                sample_features = []
                
                # Extract the closing prices across the sequence
                close_prices = np.array([X_copy[i, j*n_features] for j in range(seq_length)])
                
                # Calculate price differences between consecutive time steps
                price_diffs = np.diff(close_prices)
                
                # Calculate direction changes and streaks
                directions = np.sign(price_diffs)
                
                # Calculate other momentum indicators
                roc = (close_prices[-1] / close_prices[0] - 1) * 100
                momentum = np.sum(directions)
                volatility = np.std(price_diffs)
                
                # Add these as features
                sample_features.extend([
                    roc,                             # Rate of change
                    momentum,                        # Momentum score
                    volatility,                      # Volatility
                    np.count_nonzero(directions > 0),  # Number of up moves
                    np.count_nonzero(directions < 0),  # Number of down moves
                    np.max(np.where(directions > 0)[0]) if np.any(directions > 0) else 0,
                    np.max(np.where(directions < 0)[0]) if np.any(directions < 0) else 0,
                ])
                
                # Add strength of last few moves
                sample_features.extend(price_diffs[-3:])  # Last 3 price differences
                
                dir_features.append(sample_features)
            
            # Convert to numpy array and combine with original features
            dir_features = np.array(dir_features)
            X_with_dir_features = np.hstack([X_copy, dir_features])
            
            return X_with_dir_features
        
        # Add the method to the model instance
        model.create_direction_features = create_direction_features.__get__(model)
        
        # Override predict_direction to use enhanced features
        def predict_direction(self, X):
            """Predict price movement direction (1=up, 0=down)"""
            X_reshaped = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
            
            if hasattr(self, 'direction_classifier') and self.direction_classifier is not None:
                # Create enhanced features for direction prediction
                X_enhanced = self.create_direction_features(X_reshaped)
                return self.direction_classifier.predict(X_enhanced)
            else:
                # Fall back to deriving direction from regression prediction
                y_pred = self.predict(X)
                return (y_pred > 0.5).astype(int)
        
        # Add the method to the model instance
        model.predict_direction = predict_direction.__get__(model)
        
        # Save the model with all enhancements
        model_data = {
            'model': model.model,
            'sequence_length': model.sequence_length,
            'n_features': model.n_features,
            'model_type': model.model_type,
            'direction_classifier': self.direction_classifier,
            'direction_classifier_version': self.direction_classifier_version
        }
        
        # Save with joblib
        model_dir = os.path.dirname(output_path)
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model_data, output_path)
        
        print(f"Exported stock prediction model to {output_path} with direction classifier version {self.direction_classifier_version}")
        
        return model
``` 
