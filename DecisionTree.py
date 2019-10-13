#!/usr/bin/env python3
import numpy as np
import numpy.linalg
import numpy.random
import pandas as pd
#from DataClean import DataClean
from NaiveBayes import NaiveBayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn import tree
from helper import Helper
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

if __name__ == '__main__':
    # import data
    data = pd.read_csv('reddit_train.csv')
    # split data into train and test sets
    train_x, test_x, train_y, test_y = train_test_split(data['comments'], data['subreddits'], train_size=0.8,
                                                        test_size=0.2)
    print(len(train_x))
    helper = Helper()

    '''vectorizer = CountVectorizer()
    training_x = vectorizer.fit_transform(train_x)
    testing_x = vectorizer.transform(test_x)
    print(testing_x.shape)
    print(training_x)'''

    # tf idf
    tf_idf = TfidfVectorizer(max_features = 13000,stop_words = stopwords.words('english'))
    train_x_idf = tf_idf.fit_transform(train_x)
    test_x_idf = tf_idf.transform(test_x)
    print(train_x[1:2])

    # normalize
    #train_x_normalize, test_x_normalize = helper.run_pca(train_x_idf, test_x_idf)
    train_x_normalize = normalize(train_x_idf)
    test_x_normalize = normalize(test_x_idf)
 
    # try decision tree model
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_x_normalize, train_y)

    # predict
    clf_pred = clf.predict(test_x_normalize)
    print(clf_pred)

    # evaluation
    print(metrics.accuracy_score(test_y, clf_pred))
    print(metrics.classification_report(test_y, clf_pred))

    # plot
    # tree.plot_tree(clf.fit(training_x, train_y))
