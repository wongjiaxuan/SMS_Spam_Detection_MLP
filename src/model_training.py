"""
Model training module for MLP Classifier
"""

from sklearn.neural_network import MLPClassifier
import time
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

class SpamDetector:
    """
    MLP-based spam detection classifier
    """
    
    def __init__(self, hidden_layer_sizes=(128, 64, 32, 16)):
        """
        Initialize MLP classifier
        
        Parameters:
            hidden_layer_sizes (tuple): Architecture of hidden layers
        """
        # NOTE: class_weight is not a parameter in newer sklearn versions
        # We'll handle class imbalance by using sample weights during training
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            max_iter=100,
            batch_size=64,
            random_state=42,
            verbose=False,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        self.training_time = 0
        self.history = {'loss': [], 'validation_score': []}
        self.sample_weights = None
        
    def _compute_sample_weights(self, y_train):
        """
        Compute sample weights for handling class imbalance
        
        Parameters:
            y_train (np.ndarray): Training labels
            
        Returns:
            np.ndarray: Sample weights
        """
        # Get unique classes and their counts
        classes = np.unique(y_train)
        
        # Compute class weights using sklearn
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        
        # Create weight mapping
        weight_map = dict(zip(classes, class_weights))
        
        # Assign weights to each sample
        sample_weights = np.array([weight_map[label] for label in y_train])
        
        return sample_weights
    
    def train(self, X_train, y_train):
        """
        Train the MLP classifier
        
        Parameters:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            
        Returns:
            MLPClassifier: Trained model
        """
        print("\n" + "=" * 60)
        print("TRAINING MLP CLASSIFIER")
        print("=" * 60)
        
        print(f"Architecture: {X_train.shape[1]} -> {' -> '.join(map(str, self.model.hidden_layer_sizes))} -> 1")
        print(f"Max iterations: {self.model.max_iter}")
        print(f"Batch size: {self.model.batch_size}")
        
        # Compute sample weights for class imbalance
        self.sample_weights = self._compute_sample_weights(y_train)
        
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\nClass distribution in training set:")
        for cls, count in zip(unique, counts):
            print(f"  Class {cls}: {count} samples")
        print(f"\nUsing class weights to handle imbalance...")
        
        start_time = time.time()
        
        # Train with sample weights
        self.model.fit(X_train, y_train, sample_weight=self.sample_weights)
        
        self.training_time = time.time() - start_time
        
        # Store training history
        if hasattr(self.model, 'loss_curve_'):
            self.history['loss'] = self.model.loss_curve_
        
        print(f"\nTraining completed in {self.training_time:.2f} seconds")
        print(f"Final loss: {self.model.loss_:.4f}")
        print(f"Number of iterations: {self.model.n_iter_}")
        
        return self.model
    
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Predictions
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Parameters:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Probability scores
        """
        return self.model.predict_proba(X)
    
    def get_params(self):
        """Get model parameters"""
        return {
            'hidden_layer_sizes': self.model.hidden_layer_sizes,
            'activation': self.model.activation,
            'solver': self.model.solver,
            'max_iter': self.model.max_iter,
            'batch_size': self.model.batch_size,
            'training_time': self.training_time,
            'final_loss': self.model.loss_ if hasattr(self.model, 'loss_') else None,
            'n_iterations': self.model.n_iter_ if hasattr(self.model, 'n_iter_') else None
        }
    
    def save_model(self, filepath='models/spam_detector_model.pkl'):
        """
        Save trained model using joblib
        
        Parameters:
            filepath (str): Path to save model
        """
        import joblib
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/spam_detector_model.pkl'):
        """
        Load trained model
        
        Parameters:
            filepath (str): Path to model file
        """
        import joblib
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return self.model