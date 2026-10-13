# AutoSupport - Automated Customer Support Ticketing System

This is an end-to-end automated customer support ticketing system built with a Machine Learning classification pipeline, SQLite database integration, unit testing, and a multi-page Streamlit web dashboard.

## Features
- **Machine Learning Classification**: Categorizes and prioritizes tickets using TF-IDF and Logistic Regression pipelines.
- **SQLite Integration**: Manages ticket records with robust CRUD operations.
- **Admin Dashboard**: Live management, status updating, and filtering of support tickets.
- **Insights Dashboard**: Visual breakdown of ticket distributions.

## Setup Instructions
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Initialize Database and Train Models:
   ```bash
   python train_model.py
   python database.py
   ```
3. Run tests:
   ```bash
   python -m unittest discover -s tests
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
