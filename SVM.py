#!/usr/bin/env python3
import numpy as np
import numpy.linalg
import numpy.random
import pandas as pd
from sklearn.linear_model.tests.test_passive_aggressive import random_state
from preprocess import Preprocess
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2 
from sklearn import model_selection
import time

if __name__ == '__main__':
    # import data
    data = pd.read_csv('reddit_train.csv')
    val = pd.read_csv('reddit_test.csv')

    # store data in 
    val_x = val['comments']
    X = data['comments']
    y = data['subreddits']

    # tf idf
    tf_idf = TfidfVectorizer(sublinear_tf=True, max_df=0.03, min_df=3, norm='l2', ngram_range=(1, 2), encoding='latin-1', stop_words='english')
    X = tf_idf.fit_transform(X)
    val_x = tf_idf.transform(val_x)

    # normalize
    X = normalize(X)
    val_x = normalize(val_x)
    
    # Two features with highest chi-squared statistics are selected 
    chi2_features = SelectKBest(chi2, k=15500)
    X_kbest_features = chi2_features.fit_transform(X,y)
    val_kbest_features = chi2_features.transform(val_x)

    train_x_normalize, test_x_normalize, train_y, test_y = train_test_split(X_kbest_features, y, train_size=0.8,
                                                                    test_size=0.2)
    
    clf = SGDClassifier(loss='modified_huber',random_state=42, max_iter=3000, tol=1e-3, n_jobs=1)
    
    acc = model_selection.cross_val_score(clf, X_kbest_features, y, cv=5, scoring='accuracy')
    print("accuracy", acc.mean())
    
    start_time = time.time()
    clf.fit(train_x_normalize, train_y)
    print("--- %s runtime in seconds ---" % (time.time() - start_time))
    # predict
    clf_pred = clf.predict(test_x_normalize)
    val_pred = clf.predict(val_kbest_features)

    #write to validation file
    df = pd.DataFrame({'Category': val_pred})
    df.to_csv(index=True, path_or_buf='validation.csv', index_label='Id')

    # evaluation on testt set
    print(metrics.accuracy_score(test_y, clf_pred))
    print(metrics.classification_report(test_y, clf_pred))

