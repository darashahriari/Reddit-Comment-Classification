#!/usr/bin/env python3
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn import metrics
from nltk.corpus import stopwords
from helper import Helper
from sklearn import model_selection

if __name__ == '__main__':

    #helper functions
    helper = Helper()

    # import data
    data = pd.read_csv('data/reddit_train.csv')
    val = pd.read_csv('data/reddit_test.csv')
    x = data['comments']
    y = data['subreddits']
    val_x = val['comments']

    #tf_idf
    tf_idf = TfidfVectorizer(sublinear_tf=True, max_df=0.05, min_df=2, norm='l2', ngram_range=(1, 2), encoding='latin-1', stop_words='english')
    x = tf_idf.fit_transform(x)
    val_x = tf_idf.transform(val_x)

    # normalize
    x = normalize(x)
    val_x = normalize(val_x)

    # split data into train and test sets
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8,
                                                        test_size=0.2)
    # logistic regression
    print("start model!")
    clf = LogisticRegression(solver='lbfgs', multi_class='auto')
    import time
    start_time = time.time()
    clf.fit(train_x, train_y)
    end = time.time()
    print("time = ", end-start_time)

    # predict
    clf_pred = clf.predict(test_x)
    val_pred = clf.predict(val_x)
    helper.generate_prediction_csv(val_pred)

    # evaluation on testt set
    acc = model_selection.cross_val_score(clf, x, y, cv=5, scoring='accuracy')
    print("accuracy", acc.mean())
    print(metrics.classification_report(test_y, clf_pred))
