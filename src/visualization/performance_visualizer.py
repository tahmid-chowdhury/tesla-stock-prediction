import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

class PerformanceVisualizer:
    def __init__(self, metrics_file, output_dir="results"):
        """
        Initialize the performance visualizer
        
        Parameters:
        - metrics_file: Path to the CSV file with training metrics
        - output_dir: Directory to save visualization outputs
        """
        self.metrics_file = metrics_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_metrics(self):
        """Load metrics from CSV file"""
        try:
            return pd.read_csv(self.metrics_file)
        except Exception as e:
            print(f"Error loading metrics file: {e}")
            return None
    
    def visualize_metrics_over_time(self):
        """Visualize metrics improvements over time"""
        metrics_df = self.load_metrics()
        if metrics_df is None or metrics_df.empty:
            print("No metrics data available")
            return
        
        # Convert timestamp to datetime
        metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
        
        # Create multi-plot figure
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Plot regression metrics over time
        ax1 = axes[0, 0]
        ax1.plot(metrics_df['timestamp'], metrics_df['val_r2'], 'o-', color='blue', label='Validation R²')
        if not all(metrics_df['test_r2'] == -1):
            ax1.plot(metrics_df['timestamp'], metrics_df['test_r2'], 'o-', color='green', label='Test R²')
        ax1.set_title('R² Score Over Training Runs', fontsize=16)
        ax1.set_ylabel('R² Score', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12)
        
        # Format x-axis for better readability
        ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Plot MSE over time
        ax2 = axes[0, 1]
        ax2.plot(metrics_df['timestamp'], metrics_df['val_mse'], 'o-', color='red', label='Validation MSE')
        if not all(metrics_df['test_mse'] == -1):
            ax2.plot(metrics_df['timestamp'], metrics_df['test_mse'], 'o-', color='orange', label='Test MSE')
        ax2.set_title('Mean Squared Error Over Training Runs', fontsize=16)
        ax2.set_ylabel('MSE', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)
        
        # Format x-axis for better readability
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Plot classification metrics over time
        ax3 = axes[1, 0]
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        colors = ['blue', 'green', 'red', 'purple']
        
        for i, metric in enumerate(metrics):
            ax3.plot(metrics_df['timestamp'], metrics_df[metric], 'o-', 
                     color=colors[i], label=metric.capitalize())
        
        ax3.set_title('Classification Metrics Over Training Runs', fontsize=16)
        ax3.set_ylabel('Score', fontsize=14)
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=12)
        
        # Format x-axis for better readability
        ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Plot improvement from baseline
        ax4 = axes[1, 1]
        if len(metrics_df) > 1:
            baseline_r2 = metrics_df['val_r2'].iloc[0]
            baseline_mse = metrics_df['val_mse'].iloc[0]
            baseline_acc = metrics_df['accuracy'].iloc[0]
            baseline_f1 = metrics_df['f1_score'].iloc[0]
            
            # Calculate improvement percentages
            r2_improvement = (metrics_df['val_r2'] - baseline_r2) / abs(baseline_r2) * 100
            mse_improvement = (baseline_mse - metrics_df['val_mse']) / baseline_mse * 100
            acc_improvement = (metrics_df['accuracy'] - baseline_acc) / baseline_acc * 100
            f1_improvement = (metrics_df['f1_score'] - baseline_f1) / baseline_f1 * 100
            
            metrics_to_plot = [r2_improvement, mse_improvement, acc_improvement, f1_improvement]
            metric_names = ['R² Improvement', 'MSE Improvement', 'Accuracy Improvement', 'F1 Improvement']
            colors = ['blue', 'red', 'green', 'purple']
            
            for i, (improvement, name) in enumerate(zip(metrics_to_plot, metric_names)):
                ax4.plot(metrics_df['timestamp'], improvement, 'o-', 
                         color=colors[i], label=name)
            
            ax4.set_title('Performance Improvement from Baseline (%)', fontsize=16)
            ax4.set_ylabel('Improvement %', fontsize=14)
            ax4.grid(True, alpha=0.3)
            ax4.legend(fontsize=12)
            
            # Format x-axis for better readability
            ax4.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax4.text(0.5, 0.5, 'Need at least two training runs\nto show improvement',
                     fontsize=14, ha='center')
            ax4.axis('off')
        
        # Add timestamp to the plot
        plt.figtext(0.5, 0.01, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                   ha='center', fontsize=10)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.05)
        
        # Save the visualization
        output_path = os.path.join(self.output_dir, 'model_performance_history.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Performance visualization saved to {output_path}")
        
        # Create a summary table
        self.create_summary_table()
        
        return output_path
    
    def create_summary_table(self):
        """Create a summary table of metrics evolution"""
        metrics_df = self.load_metrics()
        if metrics_df is None or metrics_df.empty:
            return
        
        # Create summary DataFrame
        summary = pd.DataFrame(columns=['Metric', 'First Run', 'Best Run', 'Last Run', 'Improvement (%)'])
        
        # Get key metrics
        metrics = {
            'Val R²': 'val_r2',
            'Test R²': 'test_r2',
            'Val MSE': 'val_mse',
            'Test MSE': 'test_mse',
            'Accuracy': 'accuracy',
            'Precision': 'precision',
            'Recall': 'recall',
            'F1 Score': 'f1_score'
        }
        
        for metric_name, column in metrics.items():
            # Skip metrics that don't have real values
            if all(metrics_df[column] == -1):
                continue
                
            first_value = metrics_df[column].iloc[0]
            last_value = metrics_df[column].iloc[-1]
            
            # For R² and classification metrics, higher is better
            if column in ['val_r2', 'test_r2', 'accuracy', 'precision', 'recall', 'f1_score']:
                best_value = metrics_df[column].max()
                best_run = metrics_df.loc[metrics_df[column].idxmax(), 'timestamp']
                if first_value != 0:  # Avoid division by zero
                    improvement = (last_value - first_value) / abs(first_value) * 100
                else:
                    improvement = float('inf') if last_value > 0 else 0
            # For MSE, lower is better
            else:
                best_value = metrics_df[column].min()
                best_run = metrics_df.loc[metrics_df[column].idxmin(), 'timestamp']
                if first_value != 0:  # Avoid division by zero
                    improvement = (first_value - last_value) / first_value * 100
                else:
                    improvement = float('inf') if last_value < first_value else 0
            
            # Add to summary
            summary.loc[len(summary)] = [
                metric_name,
                f"{first_value:.4f}",
                f"{best_value:.4f} ({best_run})",
                f"{last_value:.4f}",
                f"{improvement:.2f}%"
            ]
        
        # Save summary to CSV
        summary_path = os.path.join(self.output_dir, 'metrics_summary.csv')
        summary.to_csv(summary_path, index=False)
        print(f"Metrics summary saved to {summary_path}")
        
        return summary
