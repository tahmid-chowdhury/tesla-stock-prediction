"""
Utility to repair corrupted JSON files.
Provides functions to recover data from malformed JSON files.
"""
import json
import os
import logging
import re
from datetime import datetime
import shutil

def repair_json_file(file_path, backup=True):
    """
    Attempt to repair a corrupted JSON file
    
    Args:
        file_path: Path to the corrupted JSON file
        backup: Whether to create a backup of the original file
        
    Returns:
        tuple: (success, data) - success is bool, data is the parsed JSON or None if failed
    """
    logging.info(f"Attempting to repair JSON file: {file_path}")
    
    # Create backup if requested
    if backup:
        backup_path = f"{file_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            shutil.copy2(file_path, backup_path)
            logging.info(f"Created backup of JSON file at: {backup_path}")
        except Exception as e:
            logging.warning(f"Failed to create backup: {e}")
    
    # Read file content as text
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except Exception as e:
        logging.error(f"Could not read JSON file: {e}")
        return False, None
    
    # Try to fix common JSON syntax issues
    try:
        # Fix 1: Try parsing as-is first
        try:
            data = json.loads(content)
            return True, data
        except json.JSONDecodeError as e:
            logging.info(f"Initial parse failed: {e}. Attempting fixes...")
        
        # Fix 2: Remove trailing commas before closing brackets
        fixed_content = re.sub(r',(\s*[\]}])', r'\1', content)
        
        # Fix 3: Add missing braces
        if not fixed_content.strip().endswith('}') and not fixed_content.strip().endswith(']'):
            if '[' in fixed_content and ']' not in fixed_content:
                fixed_content = fixed_content.rstrip() + ']'
            else:
                fixed_content = fixed_content.rstrip() + '}'
        
        # Fix 4: Handle unquoted keys
        fixed_content = re.sub(r'(\s*)([a-zA-Z0-9_]+)(\s*):', r'\1"\2"\3:', fixed_content)
        
        # Fix 5: Fix missing quotes on string values
        fixed_content = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_]*)\s*([,\]}])', r': "\1"\2', fixed_content)
        
        # Fix 6: Handle incomplete JSON by adding closing braces
        open_braces = fixed_content.count('{')
        closed_braces = fixed_content.count('}')
        if open_braces > closed_braces:
            fixed_content = fixed_content.rstrip() + ('}' * (open_braces - closed_braces))

        open_brackets = fixed_content.count('[')
        closed_brackets = fixed_content.count(']')
        if open_brackets > closed_brackets:
            fixed_content = fixed_content.rstrip() + (']' * (open_brackets - closed_brackets))
        
        # Try to parse the fixed content
        try:
            data = json.loads(fixed_content)
            
            # If successful, save the fixed content back to the file
            with open(file_path, 'w') as f:
                f.write(fixed_content)
            
            logging.info(f"Successfully repaired JSON file: {file_path}")
            return True, data
        except json.JSONDecodeError as e:
            logging.error(f"Repair attempt failed: {e}")
            
            # Last resort: create a completely new JSON with defaults
            logging.warning("Creating new JSON with default values")
            
            # For metrics file, create empty metrics with placeholder values
            new_data = create_default_json(file_path)
            
            # Save default data
            with open(file_path, 'w') as f:
                json.dump(new_data, f, indent=2)
                
            logging.info(f"Created new JSON file with default values: {file_path}")
            return True, new_data
    
    except Exception as e:
        logging.error(f"Error during JSON repair: {e}")
        return False, None

def create_default_json(file_path):
    """
    Create default JSON content based on filename
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        dict: Default JSON structure
    """
    filename = os.path.basename(file_path).lower()
    
    if 'metrics' in filename:
        # Default metrics file
        return {
            "rmse": 999.0,
            "mae": 999.0,
            "direction_accuracy": 0.5,
            "accuracy": 0.5,
            "f1": 0.5,
            "information_coefficient": 0.0
        }
    elif 'history' in filename:
        # Metrics history file
        return []
    else:
        # Generic empty object
        return {}
