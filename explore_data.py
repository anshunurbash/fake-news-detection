import pandas as pd

# Load the CSV files
df_fake = pd.read_csv('Fake.csv')
df_real = pd.read_csv('True.csv')

# Display sample data
print("Fake News Sample:")
print(df_fake.head())

print("\nReal News Sample:")
print(df_real.head())

# Check missing values
print("\nMissing Values in Fake News:\n", df_fake.isnull().sum())
print("\nMissing Values in Real News:\n", df_real.isnull().sum())

# Count of fake and real news
print("\nFake News Count:", len(df_fake))
print("Real News Count:", len(df_real))


import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
df_fake=pd.read_csv('Fake.csv')
df_real=pd.read_csv('True.csv')

df_fake['label']=1
df_real['label']=0

df=pd.concat([df_fake, df_real], ignore_index=True)

sns.countplot(x='label',data=df)
plt.title('Count of real (0) vs Fake(1) News')
plt.show()

print(df['label'].value_counts())


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Fake News WordCloud
fake_words = ' '.join(df_fake['text'].astype(str))
wordcloud_fake = WordCloud(width=800, height=500, max_font_size=110, collocations=False).generate(fake_words)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud_fake, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud for Fake News")
plt.show()

# Real News WordCloud
real_words = ' '.join(df_real['text'].astype(str))
wordcloud_real = WordCloud(width=800, height=500, max_font_size=110, collocations=False).generate(real_words)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud_real, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud for Real News")
plt.show()

print(df['label'].value_counts())

