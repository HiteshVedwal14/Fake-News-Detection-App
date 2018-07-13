import _pickle as c
import os
from sklearn import *
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import pandas as pd
from sklearn.externals import joblib

def load(clf_file):
    with open(clf_file,"rb") as fp:
        clf = c.load(fp)
    return clf


clf = load("passive_fakeNewsDetector.mdl")


inp = input(">")

#tfidf_vectorizer = TfidfVectorizer(vocabulary=c.load(open("feature.pkl", "rb")))
tfidf_vectorizer = TfidfVectorizer(decode_error='ignore')
tfidf_vectorizer =joblib.load('feature.pkl')



features=pd.Series([inp])
tfidf_test = tfidf_vectorizer.transform(features)
res = clf.predict(tfidf_test)
print(res[0])
#print(["Real","Fake!"][res[0]])

    