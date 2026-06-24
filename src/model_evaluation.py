"""
Model evaluation module
"""

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
import numpy as np
import pandas as pd

class ModelEvaluator:
    """
    Class for evaluating spam detection models
    """
    
    def __init__(self, model, X_test, y_test, label_encoder=None):
        """
        Initialize evaluator
        
        Parameters:
            model: Trained model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            label_encoder: Label encoder for class names
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.label_encoder = label_encoder
        self.metrics = {}
        self.y_pred = None
        self.y_pred_proba = None
        
    def evaluate(self):
        """
        Evaluate model performance
        
        Returns:
            dict: Performance metrics
        """
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        # Make predictions
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)
        
        # Calculate metrics
        self.metrics['accuracy'] = accuracy_score(self.y_test, self.y_pred)
        self.metrics['precision'] = precision_score(self.y_test, self.y_pred)
        self.metrics['recall'] = recall_score(self.y_test, self.y_pred)
        self.metrics['f1_score'] = f1_score(self.y_test, self.y_pred)
        
        # ROC-AUC (if binary classification)
        if self.y_pred_proba.shape[1] == 2:
            self.metrics['roc_auc'] = roc_auc_score(self.y_test, self.y_pred_proba[:, 1])
        
        # Classification report
        target_names = ['Ham', 'Spam'] if self.label_encoder is None else self.label_encoder.classes_
        self.metrics['classification_report'] = classification_report(
            self.y_test, 
            self.y_pred, 
            target_names=target_names
        )
        
        # Confusion matrix
        self.metrics['confusion_matrix'] = confusion_matrix(self.y_test, self.y_pred)
        
        # Display metrics
        self._display_metrics()
        
        return self.metrics
    
    def _display_metrics(self):
        """Display evaluation metrics"""
        print("\nPerformance Metrics:")
        print("-" * 40)
        print(f"Accuracy:  {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall:    {self.metrics['recall']:.4f}")
        print(f"F1-Score:  {self.metrics['f1_score']:.4f}")
        
        if 'roc_auc' in self.metrics:
            print(f"ROC-AUC:   {self.metrics['roc_auc']:.4f}")
        
        print("\nClassification Report:")
        print(self.metrics['classification_report'])
        
        print("\nConfusion Matrix:")
        cm = self.metrics['confusion_matrix']
        print(f"  [[TN: {cm[0,0]}, FP: {cm[0,1]}]")
        print(f"   [FN: {cm[1,0]}, TP: {cm[1,1]}]]")
    
    def get_metrics(self):
        """Get all metrics"""
        return self.metrics
    
    def get_confusion_matrix(self):
        """Get confusion matrix"""
        return self.metrics['confusion_matrix']

def calculate_metrics_df(y_test, y_pred, y_pred_proba=None):
    """
    Calculate metrics and return as DataFrame
    
    Parameters:
        y_test (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_pred_proba (np.ndarray): Prediction probabilities
        
    Returns:
        pd.DataFrame: Metrics DataFrame
    """
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }
    
    if y_pred_proba is not None and y_pred_proba.shape[1] == 2:
        metrics['ROC-AUC'] = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    return pd.DataFrame([metrics])