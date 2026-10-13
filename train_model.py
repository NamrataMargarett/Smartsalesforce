import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def train_and_evaluate():
    # 1. Load data
    data_path = 'data/complaints.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    X = df['complaint_text']
    y_category = df['category']
    y_priority = df['priority']
    
    print("Dataset loaded successfully. Shape:", df.shape)
    
    # 2. Category Model: Split, Train, Evaluate, Save
    print("\n--- Training Category Model ---")
    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
        X, y_category, test_size=0.2, random_state=42, stratify=y_category
    )
    
    category_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    category_pipeline.fit(X_train_cat, y_train_cat)
    cat_preds = category_pipeline.predict(X_test_cat)
    
    print("Category Classification Report:")
    print(classification_report(y_test_cat, cat_preds))
    
    # 3. Priority Model: Split, Train, Evaluate, Save
    print("\n--- Training Priority Model ---")
    X_train_prio, X_test_prio, y_train_prio, y_test_prio = train_test_split(
        X, y_priority, test_size=0.2, random_state=42, stratify=y_priority
    )
    
    priority_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    priority_pipeline.fit(X_train_prio, y_train_prio)
    prio_preds = priority_pipeline.predict(X_test_prio)
    
    print("Priority Classification Report:")
    print(classification_report(y_test_prio, prio_preds))
    
    # 4. Save Models
    os.makedirs('models', exist_ok=True)
    joblib.dump(category_pipeline, 'models/category_model.joblib')
    joblib.dump(priority_pipeline, 'models/priority_model.joblib')
    print("\nModels successfully saved to the 'models/' directory.")

if __name__ == '__main__':
    train_and_evaluate()
