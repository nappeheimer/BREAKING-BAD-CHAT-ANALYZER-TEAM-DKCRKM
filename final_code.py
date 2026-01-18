

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import nltk

import re

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

#JUST CLARIFYING THAT THE COMMENTS ARE NOT GPT, WE HAVE ADDED THEM FOR EASY UNDERSTADNIGN AND DEBUGGING.

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem import WordNetLemmatizer

#importing in the required libs

def preprocess(text):

  punctuations_removed_text =re.sub(r'[^a-zA-Z\s]','',text)

  #to remove punctuations and numbers from the text.

  tokens=word_tokenize(punctuations_removed_text.lower())

  filter_tokens=[token for token in tokens if token not in stopwords.words('english')]

  lemma=WordNetLemmatizer()

  l_tokens=[lemma.lemmatize(token) for token in filter_tokens]

  process_text=' '.join(l_tokens)

  return process_text
df = pd.read_csv('training.csv')
column_name = 'character'
top_n_frequent = 15
value_counts = df[column_name].value_counts()

values_to_keep = value_counts.index[:top_n_frequent]

df_filtered = df[df[column_name].isin(values_to_keep)]

df_filtered.to_csv('filtered_breakingbad_dialogue.csv', index=False)

nltk.download('all')

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



#TRAINING PART ENDS HERE.

dk = pd.read_csv('TEST WHATSAPP DATA NEW.csv')

dk = dk.sort_values(by=['sender'], ascending=True)

dk.to_csv('sorted_whatsapp_final.csv', index=False)

column_name = 'message'

min_messages= 1

min_length = 1

dk = dk[dk['message'].str.len() >= min_length]

value_counts = dk[column_name].value_counts()

values_to_remove = value_counts[value_counts < min_messages].index

dk = dk[~dk[column_name].isin(values_to_remove)]

dk = dk[dk['message'].str.len() >= min_length]

dk['message'] = dk['message'].apply(preprocess)

dk.to_csv("whatsapp_final_cleaned.csv",index = False)

X_to_predict = dk['message'].values

tfidf_test_vectors2 = tfidf_vectorizer.transform(X_to_predict)

predictions2=svm_clf.predict(tfidf_test_vectors2)

dk["predictions"] = predictions2

dk.to_csv("answer.csv",index = False)


df = pd.read_csv('answer.csv')

results = []


for sender, group in df.groupby('sender'):
    mode_prediction = group['predictions'].mode()[0]

    count_mode = (group['predictions'] == mode_prediction).sum()
    total_messages = len(group)
    accuracy = count_mode / total_messages

    results.append({
        'character': sender,
        'mode_prediction': mode_prediction,
        'accuracy': accuracy
    })

output_df = pd.DataFrame(results)

output_df.to_csv('character_predictions_analysis.csv', index=False)

print(output_df)

