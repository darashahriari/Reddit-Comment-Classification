#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import numpy.linalg
import numpy.random
import pandas as pd
from NaiveBayes import NaiveBayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB

if __name__ == '__main__':
    # cleaner = DataClean('/Users/Anna/COMP551-Project2/data/reddit_train.csv')
    # cleaner.read()
    # cleaner.partition()
    # cleaner.buildXTrain()
    # cleaner.buildYTrain(9)

    data = pd.read_csv('/Users/Anna/COMP551-Project2/data/reddit_train.csv')

    # vectorizer = CountVectorizer()
    # train_x = vectorizer.fit_transform(data['comments'])
    # train_y = vectorizer.fit_transform(data['subreddits'])
    # print(len(vectorizer.get_feature_names()))
    # print(train_x.toarray().shape[1])
    # print(train_y.toarray()[1])
    # training_x = train_x.toarray()
    # training_y = train_y.toarray()
    # num_class = len(vectorizer.get_feature_names())
    # num_feature = train_x.toarray().shape[1]

    train_x, test_x, train_y, test_y = train_test_split(data['comments'], data['subreddits'], train_size=0.8,
                                                        test_size=0.2, random_state=0)
    print(len(train_x))
    print(train_x)
    print(train_y)
    # pre
    vectorizer = CountVectorizer(max_features=1000, binary=True)
    training_x = vectorizer.fit_transform(train_x)
    # print(vectorizer.vocabulary_)
    print(len(vectorizer.get_feature_names()))
    num_feature = len(vectorizer.get_feature_names())
    print(len(vectorizer.get_feature_names()))
    print(training_x.shape)  # number of feature

    training_y = vectorizer.fit_transform(train_y)
    num_class = 20
    num_sample = len(train_x)
    feature_name = vectorizer.get_feature_names()
    print(feature_name[0])
    print(len(vectorizer.get_feature_names()))

    testing_x = vectorizer.fit_transform(test_x).toarray()
    # testing_y = vectorizer.transform(test_y)
    print(test_x.shape[0])
    print(testing_x[1][6])
    # print(testing_x)
    # print(testing_y)

    # # tf idf
    # tf_idf = TfidfVectorizer(max_features=1000, binary=True)
    # train_x_idf = tf_idf.fit_transform(train_x)
    # num_feature = len(tf_idf.get_feature_names())
    # train_y_idf = tf_idf.fit_transform(train_y).toarray()
    # num_class = 20
    # num_sample = len(train_x)
    # feature_name = tf_idf.get_feature_names()
    #
    # test_x_idf = tf_idf.transform(test_x).toarray()


    # # normalize
    # train_x_normalize = normalize(train_x_idf)
    # test_x_normalize = normalize(test_x_idf)
    # train_y_normalize = normalize(train_y_idf)
    # test_y_normalize = normalize(test_y_idf)
    # print(train_y_normalize[1])

    model_bayes = NaiveBayes(training_x=training_x,
                             training_y=training_y,
                             num_class=num_class,
                             theta_k=np.full((num_class, 1), 0.0),
                             theta_j_k=np.full((num_feature, 20), 0.0),
                             num_feature=num_feature,
                             num_sample=num_sample)
    model_bayes.fit()
    pred_y = model_bayes.predict(testing_x[1:50], feature_name)
    print(pred_y)
    print(metrics.accuracy_score(test_y[1:50], pred_y))
    print(metrics.classification_report(test_y[1:50], pred_y))
