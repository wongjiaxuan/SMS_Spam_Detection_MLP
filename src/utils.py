"""
Utility functions for SMS Spam Detection System
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime

def set_seed(seed=42):
    """
    Set random seeds for reproducibility
    
    Parameters:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def create_directory(path):
    """
    Create directory if it doesn't exist
    
    Parameters:
        path (str): Directory path to create
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def save_metrics(metrics, filename='metrics_report.txt'):
    """
    Save evaluation metrics to file
    
    Parameters:
        metrics (dict): Dictionary containing metrics
        filename (str): Output filename
    """
    with open(f'reports/{filename}', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SMS SPAM DETECTION - MODEL EVALUATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        if 'classification_report' in metrics:
            f.write("\n" + "=" * 60 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(metrics['classification_report'])

def load_dataset(filepath):
    """
    Load dataset from CSV file
    
    Parameters:
        filepath (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(filepath, encoding='latin-1')
        print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None