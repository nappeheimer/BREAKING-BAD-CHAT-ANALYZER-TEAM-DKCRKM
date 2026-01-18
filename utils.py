# utils.py
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
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer

# Download NLTK data once
nltk.download('all')

def preprocess(text):
  if not isinstance(text, str): return "" # Safety check for non-string inputs
  punctuations_removed_text =re.sub(r'[^a-zA-Z\s]','',text)
  #to remove punctuations and numbers from the text.
  tokens=word_tokenize(punctuations_removed_text.lower())
  filter_tokens=[token for token in tokens if token not in stopwords.words('english')]
  lemma=WordNetLemmatizer()
  l_tokens=[lemma.lemmatize(token) for token in filter_tokens]
  process_text=' '.join(l_tokens)
  return process_text
