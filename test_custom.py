import joblib
import numpy as np
from src.data_preprocessing import TextPreprocessor

# Load model and artifacts
model = joblib.load('models/spam_detector_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
encoder = joblib.load('models/label_encoder.pkl')

# Initialize preprocessor
preprocessor = TextPreprocessor()

# Test messages
test_messages = [
    "Congratulations! You've won a free iPhone 15. Click here to claim your prize!",
    "Hey, are we still meeting for coffee tomorrow at 3pm?",
    "URGENT: Your bank account has been compromised. Call 0800-123-456 immediately.",
    "Hi mom, I'm coming home for the holidays. Can't wait to see you!",
    "Get FREE cash now! Reply YES to claim your $1000 reward.",
    "Your Amazon order #12345 has been shipped. Track at amazon.com",
    "WINNER!! You have been selected for a £1000 prize. Call now!",
]

print("\n" + "="*60)
print("PREDICTIONS ON CUSTOM MESSAGES")
print("="*60)

for msg in test_messages:
    # Preprocess
    processed = preprocessor.preprocess(msg)
    
    # Vectorize
    features = vectorizer.transform([processed]).toarray()
    
    # Predict
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]
    
    result = encoder.inverse_transform([pred])[0]
    
    print(f"\n📝 Message: {msg}")
    print(f"   → {result.upper()} (Spam Probability: {prob:.2%})")