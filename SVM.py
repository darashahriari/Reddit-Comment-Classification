#!/usr/bin/env python3
import numpy as np
import numpy.linalg
import numpy.random
import pandas as pd
from nltk import word_tokenize, WordNetLemmatizer
from sklearn.linear_model.tests.test_passive_aggressive import random_state
from preprocess import Preprocess
from NaiveBayes import NaiveBayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize, StandardScaler, RobustScaler
from sklearn import metrics
from helper import Helper
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2 
from sklearn import model_selection

class LemmaTokenizer(object):
        def __init__(self):
            self.wnl = WordNetLemmatizer()
        def __call__(self, articles):
            return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

if __name__ == '__main__':
    # import data
    data = pd.read_csv('reddit_train.csv')
    val = pd.read_csv('reddit_test.csv')
    # split data into train and test sets
    val_x = val['comments']
    X = data['comments']
    y = data['subreddits']
    print(X)
    # processor = Preprocess()
    # preprocessedTrainingSet = Preprocess.processText(X)
    # preprocessedTestSet = Preprocess.processText(y)
    # helper = Helper()

    # tf idf
    tf_idf = TfidfVectorizer(sublinear_tf=True, max_df=0.03, min_df=3, norm='l2', ngram_range=(1, 2), encoding='latin-1', stop_words='english')
    X = tf_idf.fit_transform(X)
    val_x = tf_idf.transform(val_x)
    # ngram_range=(1, 2),
    # normalize
    X = normalize(X)
    val_x = normalize(val_x)
    
    # Two features with highest chi-squared statistics are selected 
    chi2_features = SelectKBest(chi2, k=15500)
    X_kbest_features = chi2_features.fit_transform(X,y)
    val_kbest_features = chi2_features.transform(val_x)
    print('Original feature number:', X.shape[1]) 
    print('Reduced feature number:', X_kbest_features.shape[1])
    # SVM
    train_x_normalize, test_x_normalize, train_y, test_y = train_test_split(X_kbest_features, y, train_size=0.7,
                                                                    test_size=0.3)
    # clf = SGDClassifier(loss='modified_huber',random_state=42, max_iter=3000, tol=1e-3, n_jobs=1)
    clf = MultinomialNB()
    acc = model_selection.cross_val_score(clf, X_kbest_features, y, cv=5, scoring='accuracy')
    print("accuracy", acc.mean())
    
    clf.fit(train_x_normalize, train_y)

    # predict
    clf_pred = clf.predict(test_x_normalize)
    val_pred = clf.predict(val_kbest_features)

    #write to validation file
    df = pd.DataFrame({'Category': val_pred})
    df.to_csv(index=True, path_or_buf='validation.csv', index_label='Id')

    # evaluation on testt set
    print(metrics.accuracy_score(test_y, clf_pred))
    print(metrics.classification_report(test_y, clf_pred))

