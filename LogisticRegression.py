#!/usr/bin/env python3
import numpy as np
import numpy.linalg
import numpy.random
import pandas as pd
from DataClean import DataClean
from NaiveBayes import NaiveBayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

if __name__ == '__main__':
    # import data
    data = pd.read_csv('/Users/Anna/COMP551-Project2/data/reddit_train.csv')
    test_data = pd.read_csv('/Users/Anna/COMP551-Project2/data/reddit_test.csv')

    # split data into train and test sets
    train_x, test_x, train_y, test_y = train_test_split(data['comments'], data['subreddits'], train_size=0.8,
                                                        test_size=0.2)
    print(len(train_x))
    print(data['comments'][1])

    # pre
    vectorizer = CountVectorizer(max_features=13000, stop_words=stopwords.words('english'))
    training_x = vectorizer.fit_transform(train_x)
    testing_x = vectorizer.transform(test_x)
    training_y = vectorizer.fit_transform(train_y)
    testing_y = vectorizer.transform(test_y)
    print(training_x)

    # # # for test:
    # # t_x, te_x, t_y, te_y = train_test_split(data['comments'], data['subreddits'], train_size=1,
    # #                                                     test_size=0)
    # # traintest_x = vectorizer.fit_transform(t_x)
    # # # traintest_y = vectorizer.fit_transform(t_y)
    # #
    # # final_test_x = vectorizer.transform(test_data['comments'])
    #
    # tf idf
    tf_idf = TfidfVectorizer(stop_words=stopwords.words('english'))
    train_x_idf = tf_idf.fit_transform(train_x)
    test_x_idf = tf_idf.transform(test_x)
    train_y_idf = tf_idf.fit_transform(train_y)
    test_y_idf = tf_idf.transform(test_y)
    print(train_x[1:2])

    # # normalize
    # train_x_normalize = normalize(train_x_idf)
    # test_x_normalize = normalize(test_x_idf)
    # train_y_normalize = normalize(train_y_idf)
    # test_y_normalize = normalize(test_y_idf)

    # logistic regression
    clf = LogisticRegression(solver='lbfgs', multi_class='auto')
    # lbfgs
    # clf = BernoulliNB()
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)
    # clf = KNeighborsClassifier()
    clf.fit(train_x_idf, train_y)

    # predict
    clf_pred = clf.predict(test_x_idf)
    # clf_pred = clf.predict(final_test_x)
    # f1 = open("prediction_lr", "a")
    # f1.write(str(clf_pred) + '\n')
    print(clf_pred)

    # evaluation
    print(metrics.accuracy_score(test_y, clf_pred))
    print(metrics.classification_report(test_y, clf_pred))
