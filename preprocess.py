import pandas as pd 
import string 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


stop_words=set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text=text.lower()
    text =text.translate(str.maketrans('','',string.punctuation))
    words=word_tokenize(text)
    processed_words=[stemmer.stem(word) for word in words if word not in stop_words]
    return" ".join(processed_words)
sample_text ="The Election was RIGGED! Trump says so."
processed_sample=preprocess_text(sample_text)

print("Original Text:",sample_text)
print("Processed text:",processed_sample)