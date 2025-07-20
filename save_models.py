import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import preprocess_text
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib

# Load data
df_fake = pd.read_csv('Fake.csv')
df_real = pd.read_csv('True.csv')

df_fake['label'] = 1
df_real['label'] = 0

df = pd.concat([df_fake, df_real], ignore_index=True)
df['processed_text'] = df['text'].apply(preprocess_text)

# Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_features = tfidf_vectorizer.fit_transform(df['processed_text']).toarray()
y_labels = df['label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)

# Train models
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

# Save models
joblib.dump(logistic_model, 'logistic_model.pkl')
joblib.dump(xgb_model, 'xgboost_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

print("Models and vectorizer saved successfully!")
