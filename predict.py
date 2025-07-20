import joblib
from preprocess import preprocess_text

# Load saved models and vectorizer
logistic_model = joblib.load('logistic_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_news(text):
    # Preprocess the text
    processed = preprocess_text(text)
    # Vectorize the preprocessed text
    vectorized = tfidf_vectorizer.transform([processed]).toarray()

    # Make predictions
    logistic_pred = logistic_model.predict(vectorized)[0]
    xgb_pred = xgb_model.predict(vectorized)[0]

    # Display results
    print(f"\nOriginal News: {text}")
    print("Logistic Regression Prediction:", "Fake" if logistic_pred == 1 else "Real")
    print("XGBoost Prediction:", "Fake" if xgb_pred == 1 else "Real")

# Example usage
news_input = input("Enter news article text: ")
predict_news(news_input)
