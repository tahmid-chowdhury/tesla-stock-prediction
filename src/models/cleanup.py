import os
import shutil
import argparse
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def cleanup_model_files(models_dir, keep_best=True, keep_latest=False, dry_run=False):
    """
    Clean up model files in the models directory
    
    Args:
        models_dir: Path to models directory
        keep_best: Whether to keep lstm_best.keras file
        keep_latest: Whether to keep the most recently modified model file
        dry_run: If True, only show what would be deleted without actually deleting
    
    Returns:
        Number of files deleted
    """
    if not os.path.exists(models_dir):
        logging.warning(f"Models directory {models_dir} does not exist. Nothing to clean.")
        return 0
    
    # Get all .keras files
    keras_files = [os.path.join(models_dir, f) for f in os.listdir(models_dir) 
                 if f.endswith('.keras') and os.path.isfile(os.path.join(models_dir, f))]
    
    if not keras_files:
        logging.info(f"No model files found in {models_dir}")
        return 0
    
    # Determine which files to keep
    files_to_keep = []
    
    # Keep the best model file
    best_model_path = os.path.join(models_dir, 'lstm_best.keras')
    if keep_best and os.path.exists(best_model_path):
        files_to_keep.append(best_model_path)
        logging.info(f"Keeping best model: {best_model_path}")
    
    # Keep the latest model file
    if keep_latest and keras_files:
        latest_file = max(keras_files, key=os.path.getmtime)
        if latest_file not in files_to_keep:
            files_to_keep.append(latest_file)
            logging.info(f"Keeping latest model: {latest_file}")
    
    # Delete files
    deleted_count = 0
    for file_path in keras_files:
        if file_path not in files_to_keep:
            if dry_run:
                logging.info(f"Would delete: {file_path}")
            else:
                try:
                    os.remove(file_path)
                    logging.info(f"Deleted: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    logging.error(f"Failed to delete {file_path}: {e}")
    
    action_str = "Would delete" if dry_run else "Deleted"
    logging.info(f"{action_str} {deleted_count} model files")
    return deleted_count

def cleanup_tuner_files(tuner_dir, keep_latest=False, dry_run=False):
    """
    Clean up tuner directories and files
    
    Args:
        tuner_dir: Path to tuner directory
        keep_latest: Whether to keep the most recent tuning session
        dry_run: If True, only show what would be deleted without actually deleting
    
    Returns:
        Number of directories/files deleted
    """
    if not os.path.exists(tuner_dir):
        logging.warning(f"Tuner directory {tuner_dir} does not exist. Nothing to clean.")
        return 0
    
    # Get all tuning directories
    tuning_dirs = []
    param_files = []
    
    for item in os.listdir(tuner_dir):
        item_path = os.path.join(tuner_dir, item)
        if os.path.isdir(item_path) and "tuning_" in item:
            tuning_dirs.append(item_path)
        elif item.endswith('.txt') and "best_params_" in item:
            param_files.append(item_path)
    
    if not tuning_dirs and not param_files:
        logging.info(f"No tuner files found in {tuner_dir}")
        return 0
    
    # Determine which to keep
    dirs_to_keep = []
    files_to_keep = []
    
    if keep_latest and tuning_dirs:
        # Get the latest tuning directory by modification time
        latest_dir = max(tuning_dirs, key=os.path.getmtime)
        dirs_to_keep.append(latest_dir)
        logging.info(f"Keeping latest tuning directory: {latest_dir}")
        
        # Also keep the corresponding parameter file
        latest_timestamp = os.path.basename(latest_dir).replace("tuning_", "")
        for param_file in param_files:
            if latest_timestamp in param_file:
                files_to_keep.append(param_file)
                logging.info(f"Keeping corresponding parameter file: {param_file}")
    
    # Delete directories
    deleted_count = 0
    for dir_path in tuning_dirs:
        if dir_path not in dirs_to_keep:
            if dry_run:
                logging.info(f"Would delete directory: {dir_path}")
            else:
                try:
                    shutil.rmtree(dir_path)
                    logging.info(f"Deleted directory: {dir_path}")
                    deleted_count += 1
                except Exception as e:
                    logging.error(f"Failed to delete directory {dir_path}: {e}")
    
    # Delete parameter files
    for file_path in param_files:
        if file_path not in files_to_keep:
            if dry_run:
                logging.info(f"Would delete: {file_path}")
            else:
                try:
                    os.remove(file_path)
                    logging.info(f"Deleted: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    logging.error(f"Failed to delete {file_path}: {e}")
    
    action_str = "Would delete" if dry_run else "Deleted"
    logging.info(f"{action_str} {deleted_count} tuner directories/files")
    return deleted_count

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Clean up model and tuner files')
    
    parser.add_argument('--keep-best', action='store_true', default=True,
                        help='Keep the best model file (lstm_best.keras)')
    
    parser.add_argument('--no-keep-best', action='store_false', dest='keep_best',
                        help='Do not keep the best model file')
    
    parser.add_argument('--keep-latest', action='store_true', default=False,
                        help='Keep the most recently modified model and tuner files')
    
    parser.add_argument('--models-only', action='store_true',
                        help='Clean up only model files, not tuner files')
    
    parser.add_argument('--tuner-only', action='store_true',
                        help='Clean up only tuner files, not model files')
    
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be deleted without actually deleting')
    
    args = parser.parse_args()
    
    # Get the models directory path
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    tuner_dir = os.path.join(models_dir, "tuner")
    
    logging.info(f"Starting cleanup at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_deleted = 0
    
    if not args.tuner_only:
        # Clean up model files
        model_deleted = cleanup_model_files(
            models_dir=models_dir,
            keep_best=args.keep_best, 
            keep_latest=args.keep_latest,
            dry_run=args.dry_run
        )
        total_deleted += model_deleted
    
    if not args.models_only:
        # Clean up tuner files
        tuner_deleted = cleanup_tuner_files(
            tuner_dir=tuner_dir,
            keep_latest=args.keep_latest,
            dry_run=args.dry_run
        )
        total_deleted += tuner_deleted
    
    action_str = "Would delete" if args.dry_run else "Deleted"
    logging.info(f"Cleanup complete. {action_str} a total of {total_deleted} files/directories.")

if __name__ == "__main__":
    main()
