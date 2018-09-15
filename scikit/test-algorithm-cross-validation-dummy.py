import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB

# Get a pandas DataFrame object of all the data in the csv file:
df = pd.read_csv('tweets.csv')

# Get pandas Series object of the "tweet text" column:
text = df['tweet_text']

# Get pandas Series object of the "emotion" column:
target = df['is_there_an_emotion_directed_at_a_brand_or_product']

#print(target)
#exit()

# The rows of  the "emotion" column have one of three strings:
# 'Positive emotion'
# 'Negative emotion'
# 'No emotion toward brand or product'

# Remove the blank rows from the series:
fixed_target = target[pd.notnull(text)]
fixed_text = text[pd.notnull(text)]

# Perform feature extraction:
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
count_vect.fit(fixed_text)
counts = count_vect.transform(fixed_text)

#print(counts.shape)
print(count_vect.vocabulary)
#exit()

# Train with this data with a dummy classifier:
#from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm SVC

#nb = DummyClassifier(strategy='most_frequent')

#nb = LogisticRegression()
nb = SVC()

from sklearn.model_selection import cross_val_score

scores = cross_val_score(nb, counts, fixed_target, cv=10)
print(scores)
print(scores.mean())

nb.fit(counts, target);

print(nb.predict(count_vect.transform(["I love i phone"])))

