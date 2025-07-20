import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import preprocess_text
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# Load and preprocess data (copied from vectorization.py)
df_fake = pd.read_csv('Fake.csv')
df_real = pd.read_csv('True.csv')

df_fake['label'] = 1
df_real['label'] = 0

df = pd.concat([df_fake, df_real], ignore_index=True)
df['processed_text'] = df['text'].apply(preprocess_text)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_features = tfidf_vectorizer.fit_transform(df['processed_text']).toarray()
y_labels = df['label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# XGBoost
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("\nXGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))
print("\nXGBoost Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
