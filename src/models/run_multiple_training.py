import os
import sys
import argparse
import subprocess
from datetime import datetime
import numpy as np
import random

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, project_root)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run multiple training iterations with different seeds')
    parser.add_argument('--model-type', type=str, default='direction_focused',
                       choices=['advanced', 'direction_focused'],
                       help='Type of model to train')
    parser.add_argument('--iterations', type=int, default=5, help='Number of training iterations')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs per training run')
    parser.add_argument('--base-seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--data-path', type=str, default='data/raw/TSLA.csv', help='Path to raw data')
    parser.add_argument('--refresh-data', action='store_true', help='Download fresh data')
    parser.add_argument('--compare', action='store_true', help='Compare models after training')
    
    return parser.parse_args()

def run_training_iteration(args, iteration):
    """Run a single training iteration with specific parameters"""
    # Set a different seed for each iteration
    seed = args.base_seed + iteration
    np.random.seed(seed)
    random.seed(seed)
    
    # Create timestamp for this iteration
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create iteration-specific directories
    model_dir = os.path.join(project_root, "models", f"{args.model_type}_iter{iteration}_{timestamp}")
    logs_dir = os.path.join(project_root, "logs", f"{args.model_type}_iter{iteration}_{timestamp}")
    
    # Determine which script to run
    if args.model_type == 'advanced':
        script_path = os.path.join(project_root, "src", "models", "train_advanced_model.py")
    else:
        script_path = os.path.join(project_root, "src", "models", "train_direction_focused_model.py")
    
    # Build command
    cmd = [
        sys.executable,  # Current Python interpreter
        script_path,
        f"--epochs={args.epochs}",
        f"--data-path={args.data_path}",
        f"--model-dir={model_dir}",
        f"--logs-dir={logs_dir}",
        f"--notes=Iteration {iteration}, Seed {seed}"
    ]
    
    # Add optional flags
    if args.refresh_data:
        cmd.append("--refresh-data")
    
    if args.model_type == 'direction_focused':
        cmd.append("--direction-focus")
        cmd.append("--data-augmentation")
    
    # Print command for debugging
    print(f"\nRunning iteration {iteration} with command:")
    print(" ".join(cmd))
    
    # Execute command
    try:
        # Run the command and stream output
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode != 0:
            print(f"Training failed with return code {process.returncode}")
            return False
        
        print(f"Training iteration {iteration} completed successfully")
        return True
    
    except Exception as e:
        print(f"Error running training: {e}")
        return False

def compare_trained_models(model_paths):
    """Compare the trained models"""
    # Import here to avoid circular imports
    from src.evaluation.model_version_comparison import compare_model_versions
    
    # Create comparison output directory
    comparison_dir = os.path.join(project_root, "results", "model_comparison", 
                                  f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Run comparison
    print("\nComparing trained models...")
    results_df = compare_model_versions(model_paths, comparison_dir)
    
    if results_df is not None:
        # Display comparison results
        print("\nModel Comparison Results:")
        print(results_df[['model_name', 'r2', 'f1', 'accuracy', 'precision', 'recall']].to_string())
        
        # Show best model for F1 score
        best_f1_model = results_df.loc[results_df['f1'].idxmax()]
        print(f"\nBest F1 model: {best_f1_model['model_name']} (F1: {best_f1_model['f1']:.4f})")
        
        # Copy best model to standard location
        import shutil
        best_model_path = best_f1_model['model_path']
        target_path = os.path.join(project_root, "models", "best_direction_model.joblib")
        shutil.copy(best_model_path, target_path)
        print(f"Best model copied to: {target_path}")
    
    return comparison_dir

def main():
    """Run multiple training iterations"""
    args = parse_arguments()
    
    print(f"Starting {args.iterations} training iterations of {args.model_type} model")
    start_time = datetime.now()
    
    # List to store paths of successfully trained models
    trained_models = []
    
    # Run iterations
    for i in range(args.iterations):
        print(f"\n===== Starting Training Iteration {i+1}/{args.iterations} =====")
        
        # Run training
        success = run_training_iteration(args, i)
        
        if success:
            # Find and add the model path
            if args.model_type == 'advanced':
                model_path = os.path.join(project_root, "models", f"advanced_model_iter{i}.joblib")
            else:
                model_path = os.path.join(project_root, "models", f"advanced_model_run{i+1}.joblib")
            
            if os.path.exists(model_path):
                trained_models.append(model_path)
    
    # Print summary
    end_time = datetime.now()
    total_time = end_time - start_time
    
    print(f"\n===== Multi-Training Summary =====")
    print(f"Completed {len(trained_models)}/{args.iterations} training iterations")
    print(f"Total time: {total_time}")
    
    # Compare models if requested
    if args.compare and len(trained_models) > 1:
        comparison_dir = compare_trained_models(trained_models)
        print(f"Comparison results saved to: {comparison_dir}")
    
if __name__ == "__main__":
    main()
