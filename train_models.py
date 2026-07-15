"""
Model Training Script for SMS Spam Detection
Author: 230411100058
Description: Train multiple ML models with hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from joblib import dump
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Preprocessing function
def preprocess_text(text):
    """Preprocess SMS text for model training"""
    if not text or not isinstance(text, str):
        return ""
    
    lemmatizer = WordNetLemmatizer()
    cleaned = re.sub('[^a-zA-Z0-9]', ' ', text)
    cleaned = cleaned.lower()
    words = cleaned.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

print("=" * 60)
print("SMS SPAM DETECTION - MODEL TRAINING")
print("=" * 60)

# Step 1: Load and prepare dataset
print("\n[1/5] Loading dataset...")
df = pd.read_csv('dataset/spam.csv', encoding='latin-1')

# Keep only relevant columns (v1 = label, v2 = message)
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print(f"   Total messages: {len(df)}")
print(f"   Spam messages: {len(df[df['label'] == 'spam'])}")
print(f"   Ham messages: {len(df[df['label'] == 'ham'])}")

# Step 2: Preprocess text data
print("\n[2/5] Preprocessing text data...")
df['processed_message'] = df['message'].apply(preprocess_text)

# Remove empty messages after preprocessing
df = df[df['processed_message'].str.strip() != '']

print(f"   Messages after preprocessing: {len(df)}")

# Step 3: Split data into training and testing sets
print("\n[3/5] Splitting data (80% train, 20% test)...")
X = df['processed_message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# Step 4: Define models and hyperparameters
print("\n[4/5] Training models with hyperparameter tuning...")

models = {
    'Logistic Regression': {
        'model': LogisticRegression(solver='liblinear', max_iter=1000),
        'params': {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_features': [3000, 5000],
            'model__C': [0.1, 1, 10]
        }
    },
    'SVM': {
        'model': SVC(probability=True, kernel='linear'),
        'params': {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_features': [3000, 5000],
            'model__C': [0.1, 1, 10]
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_features': [3000, 5000],
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 20]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_features': [3000, 5000],
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.1, 0.01]
        }
    },
    'Multinomial NB': {
        'model': MultinomialNB(),
        'params': {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_features': [3000, 5000],
            'model__alpha': [0.1, 0.5, 1.0]
        }
    }
}

# Train each model
results = {}

for name, config in models.items():
    print(f"\n   Training {name}...")
    
    # Create pipeline: TF-IDF + Model
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('model', config['model'])
    ])
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        config['params'],
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Save model
    model_filename = f"models/{name.replace(' ', '_').lower()}_model.joblib"
    dump(best_model, model_filename)
    print(f"      Model saved: {model_filename}")
    
    # Step 5: Evaluate model
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_df['accuracy'] = accuracy
    report_filename = f"reports/{name.replace(' ', '_').lower()}_report.csv"
    report_df.to_csv(report_filename, index=True)
    
    # Save confusion matrix
    cm_df = pd.DataFrame(
        cm,
        index=['Actual_Ham', 'Actual_Spam'],
        columns=['Predicted_Ham', 'Predicted_Spam']
    )
    cm_filename = f"reports/{name.replace(' ', '_').lower()}_confusion_matrix.csv"
    cm_df.to_csv(cm_filename, index=True)
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'best_params': grid_search.best_params_
    }
    
    print(f"      Accuracy: {accuracy:.4f}")
    print(f"      Best params: {grid_search.best_params_}")

# Step 6: Summary
print("\n" + "=" * 60)
print("[5/5] TRAINING COMPLETE - SUMMARY")
print("=" * 60)

# Sort by accuracy
sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

for i, (name, metrics) in enumerate(sorted_results, 1):
    print(f"{i}. {name}: {metrics['accuracy']:.4f}")

print("\nBest Model:", sorted_results[0][0])
print(f"Accuracy: {sorted_results[0][1]['accuracy']:.4f}")

print("\n" + "=" * 60)
print("All models and reports saved successfully!")
print("=" * 60)
