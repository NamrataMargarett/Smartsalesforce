import os
import joblib

CATEGORY_MODEL_PATH = 'models/category_model.joblib'
PRIORITY_MODEL_PATH = 'models/priority_model.joblib'

# Global placeholders for loaded pipelines
_category_model = None
_priority_model = None

def load_models():
    """Loads the TF-IDF pipelines safely. Falls back to None on failure."""
    global _category_model, _priority_model
    
    # Try to load Category Model
    if os.path.exists(CATEGORY_MODEL_PATH):
        try:
            _category_model = joblib.load(CATEGORY_MODEL_PATH)
        except Exception as e:
            print(f"Error loading Category model: {e}. Fallback enabled.")
            _category_model = None
    else:
        print("Category model file not found. Fallback enabled.")
        _category_model = None

    # Try to load Priority Model
    if os.path.exists(PRIORITY_MODEL_PATH):
        try:
            _priority_model = joblib.load(PRIORITY_MODEL_PATH)
        except Exception as e:
            print(f"Error loading Priority model: {e}. Fallback enabled.")
            _priority_model = None
    else:
        print("Priority model file not found. Fallback enabled.")
        _priority_model = None

# Initialize models on import
load_models()

def predict_ticket_attributes(complaint_text):
    """
    Predicts both category and priority of a complaint.
    Uses fallback default predictions if models are not loaded.
    """
    global _category_model, _priority_model
    
    # Default fallback values
    category = 'Technical Support'
    priority = 'Medium'
    
    # Run category prediction if available
    if _category_model is not None:
        try:
            category = str(_category_model.predict([complaint_text])[0])
        except Exception as e:
            print(f"Prediction error on category model: {e}. Using default fallback.")
            
    # Run priority prediction if available
    if _priority_model is not None:
        try:
            priority = str(_priority_model.predict([complaint_text])[0])
        except Exception as e:
            print(f"Prediction error on priority model: {e}. Using default fallback.")
            
    return category, priority

if __name__ == '__main__':
    # Simple test suite for verifying functionality
    print("--- Testing ml_utils.py ---")
    test_text_1 = "I was billed twice for my subscription this month."
    test_text_2 = "My app crashes every time I open it!"
    
    cat1, pri1 = predict_ticket_attributes(test_text_1)
    cat2, pri2 = predict_ticket_attributes(test_text_2)
    
    print(f"Test 1 Text: '{test_text_1}' -> Predicted Category: {cat1}, Priority: {pri1}")
    print(f"Test 2 Text: '{test_text_2}' -> Predicted Category: {cat2}, Priority: {pri2}")
