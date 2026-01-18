# train_model.py
import pandas as pd
import joblib 
from utils import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# --- YOUR TRAINING CODE START ---
df = pd.read_csv('training.csv') 
column_name = 'character'
top_n_frequent = 15
value_counts = df[column_name].value_counts()

values_to_keep = value_counts.index[:top_n_frequent]

df_filtered = df[df[column_name].isin(values_to_keep)]

df_filtered.to_csv('filtered_breakingbad_dialogue.csv', index=False)

df=pd.read_csv ("filtered_breakingbad_dialogue.csv")
#accessing the file.
df['dialogue'] = df['dialogue'].apply(preprocess)

#code above this line is to tokenize and pre process the csv file, and remove the stop words.

df.to_csv("breakingbaddialogue_cleaned_new.csv",index = False)

df['word_count'] = df['dialogue'].apply(lambda x: len(x.split()))
df = df[df['word_count'] > 4]
#this code is to filter out the rows with less than 4 words to filter out the data.

# filter out characters that appear less than 2 times
character_counts = df['character'].value_counts()
characters_to_keep = character_counts[character_counts >= 2].index
df_filtered = df[df['character'].isin(characters_to_keep)]

X_train, X_test , y_train, y_test = train_test_split(df_filtered['dialogue'].values,df_filtered['character'].values,test_size=0.05,random_state=42)

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')

tfidf_train_vectors = tfidf_vectorizer.fit_transform(X_train)

tfidf_test_vectors = tfidf_vectorizer.transform(X_test)

svm_clf = SVC(kernel='linear',class_weight='balanced')

svm_clf.fit(tfidf_train_vectors, y_train)

joblib.dump(svm_clf, 'svm_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
print("Models saved as 'svm_model.pkl' and 'tfidf_vectorizer.pkl'")
