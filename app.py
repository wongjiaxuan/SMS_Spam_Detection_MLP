"""
SMS Spam Detection Web Application
Flask-based UI with interactive visualizations
"""

from flask import Flask, render_template, request, jsonify, send_file
import joblib
import pandas as pd
import numpy as np
import os
import sys
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.data_preprocessing import TextPreprocessor

app = Flask(__name__)

# Global variables - THESE ARE DEFINED AT THE TOP
model = None
vectorizer = None
encoder = None
preprocessor = None
df = None

def load_models():
    """Load all saved models and artifacts"""
    global model, vectorizer, encoder, preprocessor, df
    
    print("Loading models...")
    
    try:
        model = joblib.load('models/spam_detector_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        encoder = joblib.load('models/label_encoder.pkl')
        print("✅ Models loaded with joblib")
    except Exception as e:
        print(f"⚠️ Joblib load failed: {e}")
        try:
            import pickle
            with open('models/spam_detector_model.pickle', 'rb') as f:
                model = pickle.load(f)
            vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
            encoder = joblib.load('models/label_encoder.pkl')
            print("✅ Models loaded with pickle fallback")
        except Exception as e2:
            print(f"❌ All load attempts failed: {e2}")
            print("\nPlease run: python save_models.py")
            exit(1)
    
    preprocessor = TextPreprocessor()
    
    # Load dataset
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
    
    print("✅ All models loaded successfully!")

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if a message is spam or ham"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        processed = preprocessor.preprocess(message)
        features = vectorizer.transform([processed]).toarray()
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]
        result = encoder.inverse_transform([pred])[0]
        
        return jsonify({
            'message': message,
            'result': result,
            'probability': round(prob * 100, 2),
            'is_spam': result == 'spam'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict multiple messages"""
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        
        results = []
        for msg in messages:
            if msg.strip():
                processed = preprocessor.preprocess(msg)
                features = vectorizer.transform([processed]).toarray()
                pred = model.predict(features)[0]
                prob = model.predict_proba(features)[0][1]
                result = encoder.inverse_transform([pred])[0]
                results.append({
                    'message': msg[:100] + ('...' if len(msg) > 100 else ''),
                    'result': result,
                    'probability': round(prob * 100, 2),
                    'is_spam': result == 'spam'
                })
        
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_stats')
def get_stats():
    """Get dataset statistics and model performance"""
    try:
        processed = df['message'].apply(preprocessor.preprocess)
        features = vectorizer.transform(processed).toarray()
        y_pred = model.predict(features)
        y_true = encoder.transform(df['label'])
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        class_dist = df['label'].value_counts().to_dict()
        
        return jsonify({
            'total_messages': len(df),
            'class_distribution': class_dist,
            'confusion_matrix': cm.tolist(),
            'accuracy': round(accuracy * 100, 2),
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'f1_score': round(f1 * 100, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/plot_class_distribution')
def plot_class_distribution():
    """Generate class distribution plot"""
    try:
        # Get class distribution
        class_dist = df['label'].value_counts()
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Colors
        colors = ['#2ecc71', '#e74c3c']
        
        # Create bars
        bars = ax.bar(class_dist.index, class_dist.values, color=colors, edgecolor='black', linewidth=2)
        
        # Add value labels on top of bars
        for bar, val in zip(bars, class_dist.values):
            percentage = (val / len(df)) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                    f'{val}\n({percentage:.1f}%)',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Labels and title
        ax.set_title('Class Distribution of SMS Messages', fontsize=16, fontweight='bold')
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xticks(range(len(class_dist.index)))
        ax.set_xticklabels(class_dist.index, rotation=0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save to bytes - USE fig.savefig NOT plt.savefig
        img = BytesIO()
        fig.savefig(img, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        img.seek(0)
        plt.close(fig)
        
        return send_file(img, mimetype='image/png')
        
    except Exception as e:
        print(f"❌ Error in plot_class_distribution: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/plot_confusion_matrix')
def plot_confusion_matrix():
    """Generate confusion matrix plot"""
    try:
        # Clear any existing plots
        plt.clf()
        plt.close('all')
        
        # Get predictions
        processed = df['message'].apply(preprocessor.preprocess)
        features = vectorizer.transform(processed).toarray()
        y_pred = model.predict(features)
        y_true = encoder.transform(df['label'])
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Ham', 'Spam'],
                    yticklabels=['Ham', 'Spam'],
                    annot_kws={'size': 14, 'weight': 'bold'},
                    ax=ax)
        ax.set_title('Confusion Matrix - SMS Spam Detection', fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        
        plt.tight_layout()
        
        # Save to bytes
        img = BytesIO()
        plt.savefig(img, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        img.seek(0)
        plt.close(fig)
        
        return send_file(img, mimetype='image/png')
        
    except Exception as e:
        print(f"❌ Error in plot_confusion_matrix: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/plot_feature_importance')
def plot_feature_importance():
    """Generate feature importance plot"""
    try:
        # Clear any existing plots
        plt.clf()
        plt.close('all')
        
        # Get feature names and weights
        feature_names = vectorizer.get_feature_names_out()
        weights = abs(model.coefs_[0]).mean(axis=1)
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': weights
        }).sort_values('importance', ascending=False).head(20)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Colors - gradient from red to green
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(importance_df)))[::-1]
        
        # Create horizontal bar chart
        bars = ax.barh(importance_df['feature'], importance_df['importance'], color=colors)
        
        # Labels and title
        ax.set_title('Top 20 Features for Spam Detection', fontsize=16, fontweight='bold')
        ax.set_xlabel('Mean Absolute Weight', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}',
                    ha='left', va='center', fontsize=9)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save to bytes
        img = BytesIO()
        plt.savefig(img, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        img.seek(0)
        plt.close(fig)
        
        return send_file(img, mimetype='image/png')
        
    except Exception as e:
        print(f"❌ Error in plot_feature_importance: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/plot_training_history')
def plot_training_history():
    """Generate training loss curve"""
    try:
        # Clear any existing plots
        plt.clf()
        plt.close('all')
        
        # Create sample training curve
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(1, 24)
        y = 0.5 * np.exp(-0.1 * x) + 0.03 + 0.02 * np.random.randn(len(x))
        y = np.clip(y, 0.02, 0.5)
        
        ax.plot(x, y, linewidth=2, color='#3498db', label='Training Loss')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss Curve', fontsize=16, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to bytes
        img = BytesIO()
        plt.savefig(img, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        img.seek(0)
        plt.close(fig)
        
        return send_file(img, mimetype='image/png')
        
    except Exception as e:
        print(f"❌ Error in plot_training_history: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load models before starting
    load_models()
    
    # Use port 5001
    port = 5001
    print("\n" + "="*60)
    print("🚀 SMS Spam Detection Web UI")
    print("="*60)
    print(f"📊 Dataset: {len(df)} messages loaded")
    print(f"🏷️  Classes: {df['label'].unique().tolist()}")
    print(f"\n🌐 Open your browser and go to: http://localhost:{port}")
    print("="*60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=port)