#!/usr/bin/env python3
import re
import numpy as np
import numpy.linalg
import numpy.random
import pandas as pd
from scipy.stats import chi2
from sklearn.feature_selection import SelectKBest

from NaiveBayes import NaiveBayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import WordNetLemmatizer, word_tokenize

if __name__ == '__main__':
    # import data
    data = pd.read_csv('/Users/Anna/COMP551-Project2/data/reddit_train.csv')
    # test_data = pd.read_csv('/Users/Anna/COMP551-Project2/data/reddit_test.csv')
    x = data['comments']
    y = data['subreddits']


    #tf_idf
    tf_idf = TfidfVectorizer(sublinear_tf=True, max_df=0.03, min_df=3, norm='l2', ngram_range=(1, 2), encoding='latin-1', stop_words=stopwords.words('english'))
    x = tf_idf.fit_transform(x)

    # normalize
    x = normalize(x)



    # # pre
    # vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
    # training_x = vectorizer.fit_transform(x)

    # split data into train and test sets
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8,
                                                        test_size=0.2)
    # logistic regression
    print("start model!")
    # clf = LogisticRegression(solver='lbfgs', multi_class='auto')
    # lbfgs
    # clf = MultinomialNB()
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)
    clf = KNeighborsClassifier(n_neighbors=236)
    clf.fit(train_x, train_y)

    # # creating list of K for KNN
    # neighbors = list(range(200, 230, 2))
    # print(neighbors)
    # # empty list that will hold cv scores
    # cv_scores = []
    #
    # # perform 10-fold cross validation
    # for k in neighbors:
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     scores = cross_val_score(knn, train_x, train_y, cv=5, scoring='accuracy')
    #     cv_scores.append(scores.mean())
    #     print(k, 1-scores.mean())
    #
    # mse = [1 - x for x in cv_scores]
    #
    # # determining best k
    # optimal_k = neighbors[mse.index(min(mse))]
    # print("The optimal number of neighbors is {}".format(optimal_k))




    # predict
    clf_pred = clf.predict(test_x)
    print(clf_pred)

    # evaluation
    print(metrics.accuracy_score(test_y, clf_pred))
    print(metrics.classification_report(test_y, clf_pred))
