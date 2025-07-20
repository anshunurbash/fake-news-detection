from django.shortcuts import render
import joblib
from . import models  # not strictly needed, but good 
from .preprocess import preprocess_text



# Load models and vectorizer
logistic_model = joblib.load('logistic_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
from preprocess import preprocess_text

def index(request):
    prediction = ''
    
    if request.method == 'POST':
        news_text = request.POST.get('news_text')
        processed_text = preprocess_text(news_text)
        vectorized_text = tfidf_vectorizer.transform([processed_text]).toarray()
        
        # Logistic Regression Prediction
        logistic_pred = logistic_model.predict(vectorized_text)[0]
        xgb_pred = xgb_model.predict(vectorized_text)[0]

        prediction = f"Logistic Regression: {'Fake' if logistic_pred==1 else 'Real'} | XGBoost: {'Fake' if xgb_pred==1 else 'Real'}"

    return render(request, 'detector/index.html', {'prediction': prediction})
