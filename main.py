import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WorldNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidVectorizer
from sklearn.linear_model import LinearRegression

import pandas as pd

nltk.download("wordnet")
def preprocess(phrase):
    cleaned_phrase = re.sub(re.compile("<.*?>"), "", phrase) # remove HTML tags
    cleaned_phrase = re.sub("[^A-Za-z0-9]+", " ", cleaned_phrase) # keep only words


    cleaned_phrase = cleaned_phrase.lower()

    tokens = nltk.word_tokenize(cleaned_phrase)
    stop_words = stopwords("english") # stop words to remove

    filtered_phrase = [word for word in tokens if word not in stop_words] # remove stop words

    lemmatizer = WorldNetLemmatizer()

    lemmed_phrase = [lemmatizer.lemmaize(word) for word in filtered_phrase]
    phrase = " ".join(lemmed_phrase)
    return phrase


df = pd.read_csv("labeled_data.csv")

data = df.copy()

data["class"] = [0 for elem in data["class"] if elem is not 0 else 1] # hate speech = 1 else 0

print(data["class"])


data.drop(["count", "hate_speech", "offensive_language", "neither"], axis=1, inplace=True)


data["preprocessed_tweet"] = preprocess(data["tweet"])
print(data["tweet"].head())

y = data["class"].values


X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, stratify=y)

print("Train data: ", X_train, y_train)
print("Test data: ", X_test, y_test)


vect = CountVectorizer(min_df=1) # min_df is min # of times a word can appear in doc

X_train_phrase_bow = vect.fit_transform(X_train["tweet"]) # change to preprocessed column
X_test_phrase_bow = vect.fit_transform(X_test["tweet"]) # change to preprocessed column

vectorize = TfidVectorizer(min_df=1)

X_train_phrase_tfidf = vectorizer.fit_transform(X_train["preprocessed_tweet"])
X_test_phrase_tfidf = vectorizer.transform(X_test["preprocessed_tweet"])


lin_reg = LinearRegression(penalty="l1")

lin_reg.fit(X_train_phrase_tfidf, y_train)

predicted_results = lin_reg.predict(X_test_phrase_tfidf)

print("Accuracy: ", accuracy_score(y_test, predicted_results))
