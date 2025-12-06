"""
Utility functions module.
Provides configuration loading, logger setup, and common helpers.
"""

import os
import sys
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to the config file.

    Returns:
        Configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration file: {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        raise


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration dictionary to a YAML file.

    Args:
        config: Configuration dictionary.
        save_path: Destination path for the YAML file.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        logger.info(f"Configuration saved to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration file: {e}")
        raise


def setup_logger(config: Optional[Dict[str, Any]] = None):
    """
    Set up logging system
    
    Args:
        config: Configuration dictionary
    """
    # Remove default handlers
    logger.remove()
    
    if config is None:
        # Default logger configuration
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )
        return
    
    logging_config = config.get('logging', {})
    
    # Console output
    if logging_config.get('console_output', True):
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=logging_config.get('level', 'INFO'),
            colorize=True
        )
    
    # File output
    log_file = logging_config.get('log_file')
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=logging_config.get('level', 'INFO'),
            rotation="10 MB",  # log rotation
            retention="7 days",  # keep logs for 7 days
            compression="zip"  # compress old logs
        )
        logger.info(f"Log file: {log_file}")


def ensure_dir(directory: str):
    """
    Ensure a directory exists, create it if necessary.

    Args:
        directory: Directory path.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], save_path: str, indent: int = 2):
    """
    Save a dictionary to a JSON file.

    Args:
        data: Data dictionary.
        save_path: Destination path.
        indent: JSON indent spaces.
    """
    try:
        ensure_dir(os.path.dirname(save_path))
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        logger.info(f"JSON saved to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON file: {e}")
        raise


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON file and return its contents as a dictionary.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Dictionary with JSON contents.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON file: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON file: {e}")
        raise


def get_file_size(file_path: str) -> str:
    """
    Get human-readable file size for a single file.

    Args:
        file_path: Path to the file.

    Returns:
        Size string in human-readable format.
    """
    if not os.path.exists(file_path):
        return "file not found"
    
    size_bytes = os.path.getsize(file_path)
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} PB"


def format_time(seconds: float) -> str:
    """
    Format seconds into a human-readable string.

    Args:
        seconds: Number of seconds.

    Returns:
        Readable time string.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def print_dict(data: Dict[str, Any], indent: int = 0):
    """
    Nicely print a nested dictionary.

    Args:
        data: Dictionary to print.
        indent: Indentation level.
    """
    for key, value in data.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_dict(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, where override_config overwrites base_config.

    Args:
        base_config: Base configuration.
        override_config: Overriding configuration.

    Returns:
        Merged configuration dictionary.
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate the required sections of the configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        True if valid, False otherwise.
    """
    required_sections = ['data', 'model', 'training']
    
    for section in required_sections:
            if section not in config:
                logger.error(f"Configuration missing required section: {section}")
            return False
    
            # Validate data configuration
    data_config = config.get('data', {})
    if 'target_column' not in data_config:
        logger.error("Configuration missing 'target_column' in data section")
        return False
    
    logger.info("Configuration validated successfully")
    return True


def get_project_root() -> str:
    """
    Return the project root directory (two levels up from utils.py).

    Returns:
        Path to project root as string.
    """
    current_file = Path(__file__).resolve()
    # assume utils.py is located in src/ directory
    project_root = current_file.parent.parent
    return str(project_root)


def list_models(models_dir: str = "models/") -> list:
    """
    List saved model directories with metadata.

    Args:
        models_dir: Models directory path.

    Returns:
        List of model information dicts.
    """
    if not os.path.exists(models_dir):
        logger.warning(f"Models directory does not exist: {models_dir}")
        return []
    
    models = []
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path):
            models.append({
                'name': item,
                'path': item_path,
                'size': get_directory_size(item_path),
                'modified': os.path.getmtime(item_path)
            })
    
    # Sort by modification time (newest first)
    models.sort(key=lambda x: x['modified'], reverse=True)
    
    return models


def get_directory_size(directory: str) -> str:
    """
    Compute total size of a directory and return human-readable string.

    Args:
        directory: Directory path.

    Returns:
        Human-readable directory size string.
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    
    return get_file_size_from_bytes(total_size)


def get_file_size_from_bytes(size_bytes: int) -> str:
    """
    Convert bytes to a human-readable string
    
    Args:
        size_bytes: number of bytes
        
    Returns:
        formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} PB"


class Timer:
    """Context manager timer for simple timing logs."""

    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = __import__('time').time()
        logger.info(f"{self.name} started...")
        return self

    def __exit__(self, *args):
        elapsed_time = __import__('time').time() - self.start_time
        logger.info(f"{self.name} finished, elapsed: {format_time(elapsed_time)}")


if __name__ == "__main__":
    # Test code
    config = load_config("config/config.yaml")
    print_dict(config)
