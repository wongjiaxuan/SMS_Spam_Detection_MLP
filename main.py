"""
Main execution script for SMS Spam Detection System
Fixed for SMS Spam Collection dataset format
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Check Python version
import platform
python_version = platform.python_version()
print(f"Python Version: {python_version}")

try:
    from src.utils import set_seed, create_directory, save_metrics, load_dataset
    from src.data_preprocessing import TextPreprocessor, explore_dataset
    from src.feature_extraction import FeatureExtractor, split_data, get_feature_importance
    from src.model_training import SpamDetector
    from src.model_evaluation import ModelEvaluator
    from src.visualization import Visualizer
except ModuleNotFoundError as e:
    print(f"\n❌ Error: {e}")
    print("\nPlease install required packages:")
    print("  pip install pandas numpy scikit-learn nltk matplotlib seaborn joblib")
    sys.exit(1)

def load_sms_dataset(filepath):
    """
    Load SMS Spam Collection dataset (tab-separated)
    
    Parameters:
        filepath (str): Path to dataset file
        
    Returns:
        pd.DataFrame: Loaded dataset with 'label' and 'message' columns
    """
    try:
        import pandas as pd
        
        # Try different delimiters and encodings
        # The SMS dataset is tab-separated with no header
        df = pd.read_csv(
            filepath, 
            sep='\t',           # Tab-separated
            header=None,        # No header row
            names=['label', 'message'],  # Column names
            encoding='utf-8',
            on_bad_lines='skip'  # Skip problematic lines
        )
        
        print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def main():
    """
    Main execution function
    """
    print("=" * 70)
    print(" SMS SPAM DETECTION SYSTEM ".center(70, "="))
    print(" Multi-Layer Perceptron with Backpropagation ".center(70, "="))
    print("=" * 70)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create directories
    create_directory('data')
    create_directory('models')
    create_directory('reports')
    create_directory('reports/figures')
    
    # ========================================================================
    # Step 1: Load Dataset
    # ========================================================================
    print("\n[STEP 1] Loading Dataset")
    print("-" * 50)
    
    # Try both possible filenames
    dataset_files = ['data/spam.csv', 'data/SMSSpamCollection', 'data/SMSSpamCollection.csv']
    df = None
    
    for filepath in dataset_files:
        if os.path.exists(filepath):
            print(f"Found dataset at: {filepath}")
            df = load_sms_dataset(filepath)
            if df is not None:
                break
    
    if df is None:
        print("\n❌ Could not load dataset. Please ensure dataset file is in the 'data' folder.")
        print("\n   Expected filenames: spam.csv or SMSSpamCollection")
        print("   Download from: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection")
        return
    
    # Explore dataset
    print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")
    
    print("\nClass Distribution:")
    class_dist = df['label'].value_counts()
    for label, count in class_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {label}: {count} ({percentage:.2f}%)")
    
    print("\nSample Messages (first 5):")
    print("-" * 40)
    for i in range(min(5, len(df))):
        print(f"[{df.iloc[i]['label']}] {df.iloc[i]['message'][:80]}...")
    
    # ========================================================================
    # Step 2: Preprocess Text
    # ========================================================================
    print("\n[STEP 2] Text Preprocessing")
    print("-" * 50)
    
    preprocessor = TextPreprocessor()
    df_processed = preprocessor.preprocess_dataframe(df)
    
    if len(df_processed) == 0:
        print("❌ Error: No messages after preprocessing!")
        return
    
    # Show sample
    print("\nSample after preprocessing:")
    for i in range(min(3, len(df_processed))):
        print(f"  [{df_processed.iloc[i]['label']}] {df_processed.iloc[i]['processed_message'][:60]}...")
    
    # ========================================================================
    # Step 3: Feature Extraction - TF-IDF
    # ========================================================================
    print("\n[STEP 3] Feature Extraction (TF-IDF)")
    print("-" * 50)
    
    extractor = FeatureExtractor(max_features=250, ngram_range=(1, 2))
    X, y = extractor.fit_transform(df_processed['processed_message'], df_processed['label'])
    
    print(f"Label Mapping: {extractor.get_label_mapping()}")
    
    # ========================================================================
    # Step 4: Split Data
    # ========================================================================
    print("\n[STEP 4] Data Splitting")
    print("-" * 50)
    
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    
    # ========================================================================
    # Step 5: Train Model
    # ========================================================================
    print("\n[STEP 5] Training MLP Classifier")
    print("-" * 50)
    
    detector = SpamDetector(hidden_layer_sizes=(128, 64, 32, 16))
    detector.train(X_train, y_train)
    
    # Get model parameters
    model_params = detector.get_params()
    print(f"\nModel Parameters:")
    for key, value in model_params.items():
        print(f"  {key}: {value}")
    
    # ========================================================================
    # Step 6: Evaluate Model
    # ========================================================================
    print("\n[STEP 6] Model Evaluation")
    print("-" * 50)
    
    evaluator = ModelEvaluator(detector.model, X_test, y_test, extractor.label_encoder)
    metrics = evaluator.evaluate()
    
    # ========================================================================
    # Step 7: Visualization
    # ========================================================================
    print("\n[STEP 7] Generating Visualizations")
    print("-" * 50)
    
    try:
        visualizer = Visualizer(output_dir='reports/figures')
        
        # Class distribution
        visualizer.plot_class_distribution(class_dist)
        
        # Confusion matrix
        cm = evaluator.get_confusion_matrix()
        visualizer.plot_confusion_matrix(cm, class_names=['Ham', 'Spam'])
        
        # Feature importance
        importance_df = get_feature_importance(extractor.vectorizer, detector.model)
        visualizer.plot_feature_importance(importance_df, n_top=20)
        
        # Training history
        visualizer.plot_training_history(detector.history)
        
        # ROC Curve
        if hasattr(detector.model, 'predict_proba'):
            y_pred_proba = detector.model.predict_proba(X_test)
            visualizer.plot_roc_curve(y_test, y_pred_proba)
        
        print("✅ Visualizations generated successfully!")
    except Exception as e:
        print(f"⚠️  Visualization warning: {e}")
        print("   Continuing without visualizations...")
    
    # ========================================================================
    # Step 8: Save Results
    # ========================================================================
    print("\n[STEP 8] Saving Results")
    print("-" * 50)
    
    try:
        import joblib
        
        # Save model and artifacts
        detector.save_model('models/spam_detector_model.pkl')
        joblib.dump(extractor.vectorizer, 'models/tfidf_vectorizer.pkl')
        joblib.dump(extractor.label_encoder, 'models/label_encoder.pkl')
        
        # Save metrics
        save_metrics(metrics, 'metrics_report.txt')
        
        # Save feature importance
        if 'importance_df' in locals():
            importance_df.to_csv('reports/feature_importance.csv', index=False)
        
        print("\n✅ All results saved successfully!")
        print(f"   - Model: models/spam_detector_model.pkl")
        print(f"   - Vectorizer: models/tfidf_vectorizer.pkl")
        print(f"   - Encoder: models/label_encoder.pkl")
        print(f"   - Metrics: reports/metrics_report.txt")
        print(f"   - Figures: reports/figures/")
        print(f"   - Feature Importance: reports/feature_importance.csv")
    except Exception as e:
        print(f"⚠️  Error saving results: {e}")
    
    # ========================================================================
    # Step 9: Summary Report
    # ========================================================================
    print("\n" + "=" * 70)
    print(" EXECUTION SUMMARY ".center(70, "="))
    print("=" * 70)
    
    print(f"""
    Dataset Size:        {len(df)} messages
    Training Samples:    {X_train.shape[0]}
    Testing Samples:     {X_test.shape[0]}
    Features Used:       {X.shape[1]}
    Model Architecture:  {X.shape[1]} -> {' -> '.join(map(str, detector.model.hidden_layer_sizes))} -> 1
    
    Performance:
    ─────────────────────────────────────────────
    Accuracy:            {metrics.get('accuracy', 0):.4f}
    Precision:           {metrics.get('precision', 0):.4f}
    Recall:              {metrics.get('recall', 0):.4f}
    F1-Score:            {metrics.get('f1_score', 0):.4f}
    {'ROC-AUC:' + f"            {metrics.get('roc_auc', 0):.4f}" if 'roc_auc' in metrics else ''}
    ─────────────────────────────────────────────
    """)
    
    print("\n✅ Project execution completed successfully!")
    return detector, extractor, metrics

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Ensure dataset is in data folder: spam.csv or SMSSpamCollection")
        print("2. Check all packages are installed: pip list")
        print("3. Try running: python -c 'import pandas; print(pandas.__version__)'")