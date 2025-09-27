import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class TicketClassifier:
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.label_encoder = None
        
    def create_ensemble_model(self):
        """Create ensemble of classifiers"""
        estimators = [
            ('nb', MultinomialNB()),
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(probability=True, random_state=42))
        ]
        
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
    def train(self, X_train, y_train, X_val, y_val):
        """Train the ensemble model"""
        print("Training ensemble model...")
        self.create_ensemble_model()
        self.ensemble.fit(X_train, y_train)
        
        # Validate
        train_score = self.ensemble.score(X_train, y_train)
        val_score = self.ensemble.score(X_val, y_val)
        
        print(f"Training Accuracy: {train_score:.4f}")
        print(f"Validation Accuracy: {val_score:.4f}")
        
        return self.ensemble
    
    def predict(self, X):
        """Make predictions"""
        return self.ensemble.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.ensemble.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Comprehensive model evaluation"""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
        
        return accuracy, report, cm
    
    def save_model(self, filepath):
        """Save trained model"""
        joblib.dump({
            'model': self.ensemble,
            'vectorizer': self.vectorizer
        }, filepath)
        print(f"Model saved to {filepath}")
