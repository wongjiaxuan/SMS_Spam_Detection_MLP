"""
Visualization module for model results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from src.utils import create_directory

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class Visualizer:
    """
    Class for creating visualizations
    """
    
    def __init__(self, output_dir='reports/figures'):
        """
        Initialize visualizer
        
        Parameters:
            output_dir (str): Directory to save figures
        """
        self.output_dir = output_dir
        create_directory(output_dir)
    
    def plot_class_distribution(self, class_dist, title="Class Distribution"):
        """
        Plot class distribution
        
        Parameters:
            class_dist (Series): Class distribution
            title (str): Plot title
        """
        plt.figure(figsize=(8, 6))
        colors = ['#2ecc71', '#e74c3c']
        ax = class_dist.plot(kind='bar', color=colors, edgecolor='black')
        
        # Add value labels
        for i, v in enumerate(class_dist.values):
            ax.text(i, v + 50, f'{v}\n({v/class_dist.sum()*100:.1f}%)', 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm, class_names=['Ham', 'Spam'], title="Confusion Matrix"):
        """
        Plot confusion matrix heatmap
        
        Parameters:
            cm (np.ndarray): Confusion matrix
            class_names (list): Class names
            title (str): Plot title
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={'size': 14, 'weight': 'bold'})
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, importance_df, n_top=20, title="Top Features for Spam Detection"):
        """
        Plot feature importance
        
        Parameters:
            importance_df (pd.DataFrame): Feature importance dataframe
            n_top (int): Number of top features to display
            title (str): Plot title
        """
        top_features = importance_df.head(n_top)
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(top_features)))[::-1]
        bars = plt.barh(top_features['feature'], top_features['importance'], color=colors)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Mean Absolute Weight', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, history, title="Training History"):
        """
        Plot training loss curve
        
        Parameters:
            history (dict): Training history
            title (str): Plot title
        """
        if 'loss' in history and history['loss']:
            plt.figure(figsize=(10, 6))
            plt.plot(history['loss'], linewidth=2, color='#3498db', label='Training Loss')
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/training_history.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_model_comparison(self, metrics_df, title="Model Performance Comparison"):
        """
        Plot model comparison bar chart
        
        Parameters:
            metrics_df (pd.DataFrame): Metrics dataframe
            title (str): Plot title
        """
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        available_metrics = [m for m in metrics if m in metrics_df.columns]
        
        plt.figure(figsize=(10, 6))
        values = metrics_df[available_metrics].values.flatten()
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(available_metrics)))
        bars = plt.bar(available_metrics, values, color=colors, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_test, y_pred_proba, title="ROC Curve"):
        """
        Plot ROC curve
        
        Parameters:
            y_test (np.ndarray): True labels
            y_pred_proba (np.ndarray): Prediction probabilities
            title (str): Plot title
        """
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_wordcloud(self, texts, title="Word Cloud of SMS Messages"):
        """
        Plot word cloud (requires wordcloud library)
        
        Parameters:
            texts (list): List of text messages
            title (str): Plot title
        """
        try:
            from wordcloud import WordCloud
            
            combined_text = ' '.join(texts)
            wordcloud = WordCloud(
                width=800, 
                height=400,
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(combined_text)
            
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/wordcloud.png', dpi=300, bbox_inches='tight')
            plt.show()
        except ImportError:
            print("WordCloud library not installed. Skipping word cloud plot.")