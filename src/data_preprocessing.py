import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

class DataPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if isinstance(text, str):
            # Convert to lowercase
            text = text.lower()
            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            # Tokenize
            words = text.split()
            # Remove stopwords and stem
            words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
            return ' '.join(words)
        return ""
    
    def prepare_features(self, df, text_columns=['subject', 'description']):
        """Prepare features for training"""
        # Combine text columns
        df['combined_text'] = df[text_columns].apply(
            lambda x: ' '.join(x.dropna().astype(str)), axis=1
        )
        
        # Clean text
        df['cleaned_text'] = df['combined_text'].apply(self.clean_text)
        
        return df
    
    def fit_vectorizer(self, texts):
        """Fit TF-IDF vectorizer"""
        self.vectorizer.fit(texts)
        
    def transform_features(self, texts):
        """Transform texts to TF-IDF features"""
        return self.vectorizer.transform(texts)
