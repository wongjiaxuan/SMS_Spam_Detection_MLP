"""
Re-save models with proper format for web app
"""

import joblib
import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import TextPreprocessor
from src.feature_extraction import FeatureExtractor
from src.model_training import SpamDetector
from src.utils import set_seed, create_directory

print("="*60)
print("RE-SAVING MODELS FOR WEB APPLICATION")
print("="*60)

# Set seed for reproducibility
set_seed(42)

# Create directories
create_directory('models')

# Load dataset
print("\n[1] Loading dataset...")
try:
    df = pd.read_csv('data/spam.csv', sep='\t', header=None, names=['label', 'message'], encoding='utf-8')
    print(f"✅ Dataset loaded: {len(df)} messages")
except:
    try:
        df = pd.read_csv('data/spam.csv', encoding='utf-8')
        if 'label' not in df.columns or 'message' not in df.columns:
            df = pd.read_csv('data/spam.csv', sep='\t', header=None, names=['label', 'message'], encoding='latin-1')
        print(f"✅ Dataset loaded: {len(df)} messages")
    except:
        df = pd.read_csv('data/spam.csv', sep='\t', header=None, names=['label', 'message'], encoding='latin-1')
        print(f"✅ Dataset loaded: {len(df)} messages")

# Preprocess
print("\n[2] Preprocessing messages...")
preprocessor = TextPreprocessor()
df_processed = preprocessor.preprocess_dataframe(df)
print(f"✅ Preprocessed: {len(df_processed)} messages")

# Feature extraction
print("\n[3] Extracting TF-IDF features...")
extractor = FeatureExtractor(max_features=250, ngram_range=(1, 2))
X, y = extractor.fit_transform(df_processed['processed_message'], df_processed['label'])
print(f"✅ Feature matrix: {X.shape}")

# Split data
from sklearn.model_selection import train_test_split
print("\n[4] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✅ Training: {X_train.shape[0]}, Testing: {X_test.shape[0]}")

# Train model
print("\n[5] Training MLP Classifier...")
detector = SpamDetector(hidden_layer_sizes=(128, 64, 32, 16))
detector.train(X_train, y_train)
print(f"✅ Training complete in {detector.training_time:.2f}s")

# Save with proper format
print("\n[6] Saving models...")

# Save using joblib with proper protocol
joblib.dump(detector.model, 'models/spam_detector_model.pkl', compress=3)
print("✅ Saved: spam_detector_model.pkl")

joblib.dump(extractor.vectorizer, 'models/tfidf_vectorizer.pkl', compress=3)
print("✅ Saved: tfidf_vectorizer.pkl")

joblib.dump(extractor.label_encoder, 'models/label_encoder.pkl', compress=3)
print("✅ Saved: label_encoder.pkl")

# Also save as pickle for compatibility
import pickle
with open('models/spam_detector_model.pickle', 'wb') as f:
    pickle.dump(detector.model, f)
print("✅ Saved: spam_detector_model.pickle")

print("\n" + "="*60)
print("✅ ALL MODELS SAVED SUCCESSFULLY!")
print("="*60)
print("\nFiles saved:")
print("  - models/spam_detector_model.pkl")
print("  - models/tfidf_vectorizer.pkl")
print("  - models/label_encoder.pkl")
print("  - models/spam_detector_model.pickle (backup)")