import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import preprocess_text


df_fake = pd.read_csv('Fake.csv')
df_real = pd.read_csv('True.csv')


df_fake['label'] = 1
df_real['label'] = 0


df = pd.concat([df_fake, df_real], ignore_index=True)


df['processed_text'] = df['text'].apply(preprocess_text)


tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_features = tfidf_vectorizer.fit_transform(df['processed_text']).toarray()
y_labels = df['label'].values


print("TF-IDF feature matrix shape:", X_features.shape)
print("Labels shape:", y_labels.shape)
