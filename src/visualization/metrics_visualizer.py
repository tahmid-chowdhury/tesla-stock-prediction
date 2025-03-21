import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging

class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

class MetricsVisualizer:
    def __init__(self, results_dir=None):
        """
        Initialize metrics visualizer
        
        Args:
            results_dir: Directory to store results/metrics
        """
        if results_dir is None:
            self.results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
        else:
            self.results_dir = results_dir
            
        self.metrics_history_file = os.path.join(self.results_dir, "metrics_history.json")
        os.makedirs(self.results_dir, exist_ok=True)
        
    def save_metrics(self, metrics, iteration=None):
        """
        Save metrics to history file
        
        Args:
            metrics: Dictionary containing metrics
            iteration: Custom iteration number (optional)
        """
        # Ensure metrics contains only JSON-serializable values
        serializable_metrics = {}
        for key, value in metrics.items():
            # Convert NumPy types to Python native types
            if isinstance(value, (np.integer, np.floating)):
                serializable_metrics[key] = float(value)
            elif isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            else:
                serializable_metrics[key] = value
        
        # Load existing metrics history
        history = self.load_metrics_history()
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # If no iteration provided, use the next number in sequence
        if iteration is None:
            iteration = len(history) + 1
            
        # Add new metrics with timestamp
        entry = {
            "iteration": iteration,
            "timestamp": timestamp,
            "metrics": serializable_metrics
        }
        
        history.append(entry)
        
        # Save updated history
        try:
            with open(self.metrics_history_file, 'w') as f:
                json.dump(history, f, indent=2, cls=NumpyJSONEncoder)
            logging.info(f"Metrics saved to {self.metrics_history_file}")
        except Exception as e:
            logging.error(f"Error saving metrics history: {e}")
            
    def load_metrics_history(self):
        """
        Load metrics history from file
        
        Returns:
            List of metrics dictionaries
        """
        if os.path.exists(self.metrics_history_file):
            try:
                with open(self.metrics_history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading metrics history: {e}")
                # If the file is corrupt, rename it and create a new one
                try:
                    backup_file = f"{self.metrics_history_file}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    os.rename(self.metrics_history_file, backup_file)
                    logging.warning(f"Renamed corrupt metrics history file to {backup_file}")
                except Exception as rename_error:
                    logging.error(f"Could not rename corrupt file: {rename_error}")
                return []
        return []
        
    def plot_training_history(self, n_iterations=10, save_path=None):
        """
        Plot metrics across training iterations
        
        Args:
            n_iterations: Number of previous iterations to plot
            save_path: Path to save the plot (if None, auto-generated)
        """
        history = self.load_metrics_history()
        
        # Ensure we have at least one iteration
        if not history:
            logging.warning("No metrics history available to plot")
            return None
            
        # Get latest n iterations
        recent_history = history[-n_iterations:] if len(history) > n_iterations else history
        
        # Extract metrics
        iterations = [entry.get("iteration", i+1) for i, entry in enumerate(recent_history)]
        timestamps = [entry.get("timestamp", "") for entry in recent_history]
        
        # Extract metrics we want to plot
        accuracy = []
        precision = []
        recall = []
        f1_score = []
        
        for entry in recent_history:
            metrics = entry.get("metrics", {})
            accuracy.append(metrics.get("accuracy", 0))
            precision.append(metrics.get("precision", 0))
            recall.append(metrics.get("recall", 0))
            f1_score.append(metrics.get("f1", 0))
            
        # Calculate mean of metrics
        mean_metrics = np.mean([accuracy, precision, recall, f1_score], axis=0)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        plt.plot(iterations, accuracy, 'o-', color='blue', label='Accuracy')
        plt.plot(iterations, precision, 'o-', color='green', label='Precision')
        plt.plot(iterations, recall, 'o-', color='red', label='Recall')
        plt.plot(iterations, f1_score, 'o-', color='purple', label='F1-Score')
        # Fix the redundant linestyle definition
        plt.plot(iterations, mean_metrics, marker='o', color='black', linestyle='--', linewidth=2, label='Mean')
        
        plt.title('Performance Metrics Across Training Iterations', fontsize=16)
        plt.xlabel('Training Iteration', fontsize=12)
        plt.ylabel('Score (0.0 - 1.0)', fontsize=12)
        
        # Set y-axis range from 0 to 1
        plt.ylim(0, 1)
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3)
        
        # Add legend
        plt.legend(loc='lower right')
        
        # Add iteration timestamps as tooltips on hover
        for i, (iter_num, ts) in enumerate(zip(iterations, timestamps)):
            plt.annotate(f"Iteration {iter_num}\n{ts}",
                        (iter_num, mean_metrics[i]),
                        xytext=(0, 30),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                        visible=False)
        
        # Generate save path if not provided
        if save_path is None:
            save_path = os.path.join(self.results_dir, 
                                     f'training_metrics_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        
        plt.tight_layout()
        plt.savefig(save_path)
        logging.info(f"Training metrics history plot saved to {save_path}")
        
        # Close the figure to free memory
        plt.close()
        
        return save_path

    def plot_interactive_metrics(self, n_iterations=10):
        """
        Create an interactive metrics visualization that allows exploring the data
        
        Args:
            n_iterations: Number of previous iterations to include
        """
        history = self.load_metrics_history()
        
        if not history:
            logging.warning("No metrics history available to plot")
            return None
        
        # Get latest n iterations
        recent_history = history[-n_iterations:] if len(history) > n_iterations else history
        
        # Convert to DataFrame for easier manipulation
        data = []
        for entry in recent_history:
            metrics = entry.get("metrics", {})
            item = {
                "iteration": entry.get("iteration", 0),
                "timestamp": entry.get("timestamp", ""),
                "accuracy": metrics.get("accuracy", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "f1_score": metrics.get("f1", 0),
            }
            # Add mean metric
            item["mean"] = np.mean([
                item["accuracy"], 
                item["precision"], 
                item["recall"], 
                item["f1_score"]
            ])
            data.append(item)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save metrics table to CSV for reference
        csv_path = os.path.join(self.results_dir, 
                                f'training_metrics_table_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        df.to_csv(csv_path, index=False)
        logging.info(f"Training metrics table saved to {csv_path}")
        
        # Return the dataframe for further analysis if needed
        return df
