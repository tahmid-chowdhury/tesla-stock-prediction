"""
Utility for monitoring and optimizing model training performance
"""
import time
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class PerformanceMonitor:
    """
    Monitor training performance and suggest optimizations
    """
    def __init__(self):
        """Initialize the performance monitor"""
        self.times = {}
        self.metrics = {}
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
        os.makedirs(self.results_dir, exist_ok=True)
    
    def start_timer(self, stage_name):
        """Start timing a specific stage of processing"""
        self.times[stage_name] = {'start': time.time(), 'end': None, 'duration': None}
        return self
    
    def stop_timer(self, stage_name):
        """Stop timing a specific stage and record duration"""
        if stage_name in self.times:
            self.times[stage_name]['end'] = time.time()
            self.times[stage_name]['duration'] = self.times[stage_name]['end'] - self.times[stage_name]['start']
        return self
    
    def record_metric(self, name, value):
        """Record a performance metric"""
        self.metrics[name] = value
        return self
    
    def log_memory_usage(self):
        """Log current memory usage"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_usage = memory_info.rss / 1024 / 1024  # Convert to MB
            self.record_metric('memory_usage_mb', memory_usage)
            logging.info(f"Current memory usage: {memory_usage:.2f} MB")
        except ImportError:
            logging.warning("psutil not installed. Cannot monitor memory usage.")
        return self
    
    def log_performance_summary(self):
        """Log a summary of all recorded performance metrics"""
        logging.info("Performance Summary:")
        logging.info("-" * 40)
        
        if self.times:
            logging.info("Timing Information:")
            for stage, time_info in self.times.items():
                if time_info['duration'] is not None:
                    if time_info['duration'] > 60:
                        # Convert to minutes for readability
                        mins = time_info['duration'] / 60
                        logging.info(f"  {stage}: {mins:.2f} minutes")
                    else:
                        logging.info(f"  {stage}: {time_info['duration']:.2f} seconds")
        
        if self.metrics:
            logging.info("Performance Metrics:")
            for name, value in self.metrics.items():
                logging.info(f"  {name}: {value}")
        
        logging.info("-" * 40)
        return self
    
    def plot_performance(self):
        """Create a visualization of performance metrics"""
        if not self.times:
            return None
        
        plt.figure(figsize=(10, 6))
        
        # Plot timing information as a bar chart
        stages = []
        durations = []
        
        for stage, time_info in self.times.items():
            if time_info['duration'] is not None:
                stages.append(stage)
                durations.append(time_info['duration'])
        
        if not stages:
            return None
        
        y_pos = np.arange(len(stages))
        plt.barh(y_pos, durations, align='center', alpha=0.5)
        plt.yticks(y_pos, stages)
        plt.xlabel('Time (seconds)')
        plt.title('Performance by Processing Stage')
        
        # Save plot
        filename = os.path.join(self.results_dir, f'performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
        logging.info(f"Performance visualization saved to {filename}")
        return filename
    
    def suggest_optimizations(self, X_train=None, history=None):
        """
        Analyze performance and suggest optimizations
        
        Args:
            X_train: Training data
            history: Training history object
        
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        # Check overall training time
        if 'model_training' in self.times and self.times['model_training']['duration'] is not None:
            training_time = self.times['model_training']['duration']
            
            if training_time > 600:  # More than 10 minutes
                suggestions.append("Training time is very long. Consider using a simplified model architecture.")
                suggestions.append("Try reducing the feature count with --reduced-features and --feature-count options.")
            
            if training_time > 300:  # More than 5 minutes
                suggestions.append("Try increasing batch size to speed up training.")
                
        # Check data size
        if X_train is not None:
            if len(X_train) > 10000:
                suggestions.append(f"Large dataset ({len(X_train)} samples). Consider using --sample-training.")
                
            if X_train.shape[2] > 30:
                suggestions.append(f"High feature dimensionality ({X_train.shape[2]} features). Try --reduced-features.")
        
        # Check convergence from history
        if history is not None and hasattr(history, 'history'):
            # Check if training was stopped early
            if len(history.history['loss']) < self.metrics.get('max_epochs', 100):
                early_stop_epoch = len(history.history['loss'])
                suggestions.append(f"Training stopped early at epoch {early_stop_epoch}. " +
                                  f"Consider reducing max_epochs to {early_stop_epoch + 5}.")
            
            # Check if validation loss is still decreasing
            if 'val_loss' in history.history:
                val_loss = history.history['val_loss']
                if len(val_loss) > 5 and val_loss[-1] < val_loss[-2] < val_loss[-3]:
                    suggestions.append("Validation loss was still decreasing. Try increasing early stopping patience.")
        
        # Check memory usage
        if 'memory_usage_mb' in self.metrics:
            memory_mb = self.metrics['memory_usage_mb']
            if memory_mb > 4000:  # More than 4GB
                suggestions.append(f"High memory usage ({memory_mb:.0f} MB). Try reducing batch size or data sampling.")
        
        # Log suggestions
        if suggestions:
            logging.info("Performance optimization suggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                logging.info(f"{i}. {suggestion}")
        else:
            logging.info("No specific performance optimizations suggested.")
            
        return suggestions
