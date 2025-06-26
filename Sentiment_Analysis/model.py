import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score , precision_score , recall_score , f1_score

# import libraries requires fro nlp
import nltk # nl toolkit
import re # regular expression

from nltk.corpus import stopwords # library for importing stopwords

#download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

df = pd.read_csv("IMDB Dataset.csv")

# Mapping the sentiments to some numerical value
df["sentiment"] = df["sentiment"].map({
    "positive" : 1,
    "negative" : 0
})

# clean the text
def clean_text(text) :
  text = re.sub(r"[^a-zA-Z]"," ",text).lower()
  tokens = text.split()
  tokens = [word for word in tokens if word not in stop_words]
  return " ".join(tokens)

# apply the clean text finction on review
df["cleaned_review"] = df["review"].apply(clean_text)

# feature extraction
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cleaned_review"])

y= df["sentiment"]

# divide the dataset into train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# make the prediction
y_pred = model.predict(X_test)

# calculate the performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", cr)

import joblib
joblib.dump(model,"sentiment_analysis.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")