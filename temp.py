import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
#from pandas_ml import ConfusionMatrix
from matplotlib import pyplot as plt
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import HashingVectorizer
import os
import itertools
import _pickle as c
from sklearn.externals import joblib

import numpy as np

def save(clf, name):
    with open(name, 'wb') as fp:
        c.dump(clf, fp)
    print("saved")

'''Importing the dataset'''

#Loading the dataset
df= pd.read_csv("fake_or_real_news.csv")
df.set_index("Unnamed: 0",inplace=True, drop=True)
#making the 1st unnamed column as the index

df.head()
y=df.label
df.drop('label', axis=1, inplace =True)
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

#Converting training and testing dataset into count vectors
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)

#Converting training and testing dataset into  tfidf vectors
tfidf_vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7) #This removes words that appear more than 70% times
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
#c.dump(tfidf_vectorizer.vocabulary_,open("feature.pkl","wb"))
joblib.dump(tfidf_vectorizer, 'feature.pkl')
tfidf_test = tfidf_vectorizer.transform(X_test)

count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

difference = set(count_df.columns) -set(tfidf_df.columns)
print(difference)
print(count_df.equals(tfidf_df))
print(count_df.head())
print(tfidf_df.head())


#Multinomial Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(tfidf_train, y_train) #fitting naive bayes classifier according to x and y
pred =clf.predict(tfidf_test) #Perform classification on an array of test vectors x.
scoreMN = metrics.accuracy_score(y_test, pred)
print("Accuracy: %0.3f" % scoreMN)



# Applying Passive Aggressive Classifier
linear_clf = PassiveAggressiveClassifier(n_iter=50)
linear_clf.fit(tfidf_train, y_train)
pred = linear_clf.predict(tfidf_test)
scorePAC = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % scorePAC)




clf = MultinomialNB(alpha=0.1)               # Additive (Laplace/Lidstone) smoothing parameter
last_score = 0
for alpha in np.arange(0,1,.1):
    nb_classifier = MultinomialNB(alpha=alpha)
    nb_classifier.fit(tfidf_train, y_train)
    pred = nb_classifier.predict(tfidf_test)
    score = metrics.accuracy_score(y_test, pred)
    if score > last_score:
        clf = nb_classifier
    print("Alpha: {:.2f} Score: {:.5f}".format(alpha, score))



print("Multinomial Naive Bayes Classifier Accuracy: %0.3f" % scoreMN)
print("Passive Aggressive Classifier Accuracy: %0.3f" % scorePAC)

save(clf,"fakeNewsDetector.mdl")
save(linear_clf,"passive_fakeNewsDetector.mdl")
