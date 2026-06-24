"""
Data preprocessing module for SMS Spam Detection
"""

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

class TextPreprocessor:
    """
    Class for preprocessing SMS text messages
    """
    
    def __init__(self):
        """Initialize preprocessor with lemmatizer and stopwords"""
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords_set = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """
        Clean text by removing HTML tags, URLs, special characters, and numbers
        
        Parameters:
            text (str): Raw input text
            
        Returns:
            str: Cleaned text
        """
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove non-ASCII characters (except periods)
        text = re.sub(r'[^\x00-\x7F.]', ' ', text)
        
        # Remove punctuation except periods
        text = re.sub(f'[{re.escape(string.punctuation.replace(".", ""))}]', '', text)
        
        # Remove isolated numbers
        text = re.sub(r'\b\d+\b', '', text)
        
        # Replace multiple periods
        text = re.sub(r'\.{2,}', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline
        
        Parameters:
            text (str): Raw input text
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and non-alphabetic tokens, then lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stopwords_set and token.isalpha()
        ]
        
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, text_column='message'):
        """
        Preprocess entire dataframe
        
        Parameters:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            
        Returns:
            pd.DataFrame: Dataframe with preprocessed column
        """
        print("Preprocessing text messages...")
        df_processed = df.copy()
        df_processed['processed_message'] = df_processed[text_column].apply(self.preprocess)
        
        # Remove empty messages
        df_processed = df_processed[df_processed['processed_message'].str.len() > 0]
        
        print(f"Preprocessed {len(df_processed)} messages")
        return df_processed

def explore_dataset(df):
    """
    Explore and display dataset statistics
    
    Parameters:
        df (pd.DataFrame): Input dataframe
    """
    print("\n" + "=" * 60)
    print("DATASET EXPLORATION")
    print("=" * 60)
    
    print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print("\nColumn Names:")
    for col in df.columns:
        print(f"  - {col}")
    
    print("\nClass Distribution:")
    class_dist = df['label'].value_counts()
    for label, count in class_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {label}: {count} ({percentage:.2f}%)")
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nSample Messages (first 5):")
    print("-" * 40)
    for i in range(min(5, len(df))):
        print(f"[{df.iloc[i]['label']}] {df.iloc[i]['message'][:80]}...")
    
    return class_dist