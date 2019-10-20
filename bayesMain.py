#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import time
from NaiveBayes import NaiveBayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics

if __name__ == '__main__':
    data = pd.read_csv('data/reddit_train.csv')
    train_x, test_x, train_y, test_y = train_test_split(data['comments'], data['subreddits'], train_size=0.8,
                                                        test_size=0.2, random_state=0)
    print(len(train_x))
    print(train_x)
    print(train_y)
    # pre
    vectorizer = CountVectorizer(binary=True)
    training_x = vectorizer.fit_transform(train_x)
    print(len(vectorizer.get_feature_names()))
    num_feature = len(vectorizer.get_feature_names())
    print(len(vectorizer.get_feature_names()))
    print(training_x.shape)  # number of feature

    testing_x = vectorizer.transform(test_x).toarray()
    training_y = vectorizer.fit_transform(train_y)
    num_class = 20
    num_sample = len(train_x)
    feature_name = vectorizer.get_feature_names()
    print(feature_name[0])
    print(len(vectorizer.get_feature_names()))

    print(test_x.shape[0])
    print(testing_x[1][6])
    print(feature_name)

    model_bayes = NaiveBayes(training_x=training_x,
                             training_y=training_y,
                             num_class=num_class,
                             theta_k=np.full((num_class, 1), 0.0),
                             theta_j_k=np.full((num_feature, num_class), 0.0),
                             num_feature=num_feature,
                             num_sample=num_sample)
    
    start_time = time.time()
    model_bayes.fit()
    print("--- %s runtime in seconds ---" % (time.time() - start_time))

    pred_y = model_bayes.predict(testing_x[1:100], feature_name)
    print(pred_y)
    print(test_y[1:100])
    print(metrics.accuracy_score(test_y[1:100], pred_y))
    print(metrics.classification_report(test_y[1:100], pred_y))
