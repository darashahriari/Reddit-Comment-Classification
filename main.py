#!/usr/bin/env python
# coding: utf-8

import sys
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

if __name__ == '__main__':
    # cleaner = DataClean('/Users/Anna/COMP551-Project2/data/reddit_train.csv')
    # cleaner.read()
    # cleaner.partition()
    # cleaner.buildXTrain()
    # cleaner.buildYTrain(9)

    data = pd.read_csv('/Users/Anna/COMP551-Project2/data/reddit_train.csv')
    vectorizer = CountVectorizer()
    train_x = vectorizer.fit_transform(data['comments'])
    train_y = vectorizer.fit_transform(data['subreddits'])
    print(len(vectorizer.get_feature_names()))
    print(train_x.toarray().shape[1])
    print(train_y.toarray()[1])
    training_x = train_x.toarray()
    training_y = train_y.toarray()
    num_class = len(vectorizer.get_feature_names())
    num_feature = train_x.toarray().shape[1]

    model_bayes = NaiveBayes(training_x=training_x,
                             training_y=training_y,
                             num_class=num_class,
                             theta_k=np.full((num_class, 1), 0.0),
                             theta_j_k=np.full((num_feature, 20), 0.0),
                             num_feature=num_feature)
    #model_bayes.fit()
