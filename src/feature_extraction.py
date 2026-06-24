"""
Feature extraction module using TF-IDF vectorization
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class FeatureExtractor:
    """
    Class for extracting TF-IDF features from text data
    """
    
    def __init__(self, max_features=250, ngram_range=(1, 2)):
        """
        Initialize TF-IDF vectorizer
        
        Parameters:
            max_features (int): Maximum number of features
            ngram_range (tuple): N-gram range
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        self.label_encoder = LabelEncoder()
        self.max_features = max_features
        
    def fit_transform(self, texts, labels=None):
        """
        Fit and transform text data
        
        Parameters:
            texts (array-like): Text messages
            labels (array-like): Labels for encoding
            
        Returns:
            tuple: (features, encoded_labels)
        """
        print("Extracting TF-IDF features...")
        X = self.vectorizer.fit_transform(texts).toarray()
        print(f"Feature matrix shape: {X.shape}")
        
        if labels is not None:
            y = self.label_encoder.fit_transform(labels)
            return X, y
        
        return X
    
    def transform(self, texts):
        """
        Transform new text data using fitted vectorizer
        
        Parameters:
            texts (array-like): Text messages
            
        Returns:
            np.ndarray: Feature matrix
        """
        return self.vectorizer.transform(texts).toarray()
    
    def get_feature_names(self):
        """Get feature names from vectorizer"""
        return self.vectorizer.get_feature_names_out()
    
    def get_label_mapping(self):
        """Get label encoding mapping"""
        return dict(zip(
            self.label_encoder.classes_,
            self.label_encoder.transform(self.label_encoder.classes_)
        ))

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets with stratification
    
    Parameters:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Labels
        test_size (float): Proportion for testing
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"\nSplitting data: {test_size*100:.0f}% test set")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set:  {X_test.shape[0]} samples")
    
    # Display class distribution in train set
    print("\nTraining set class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        percentage = (count / len(y_train)) * 100
        print(f"  Class {label}: {count} ({percentage:.2f}%)")
    
    return X_train, X_test, y_train, y_test

def get_feature_importance(vectorizer, model):
    """
    Calculate feature importance based on model weights
    
    Parameters:
        vectorizer: Fitted TF-IDF vectorizer
        model: Trained MLP model
        
    Returns:
        pd.DataFrame: Feature importance scores
    """
    feature_names = vectorizer.get_feature_names_out()
    weights = abs(model.coefs_[0]).mean(axis=1)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': weights
    }).sort_values('importance', ascending=False)
    
    return importance_df