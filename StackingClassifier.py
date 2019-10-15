#!/usr/bin/env python3
import numpy as np
import numpy.linalg
import numpy.random
import pandas as pd
#from DataClean import DataClean
from NaiveBayes import NaiveBayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn import metrics
from helper import Helper
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
    #val = pd.read_csv('reddit_test.csv')
    # split data into train and test sets
    train_x, test_x, train_y, test_y = train_test_split(data['comments'], data['subreddits'], train_size=0.8,
                                                        test_size=0.2)
    #val_x = val['comments']
    X = data['comments']
    y = data['subreddits']
    print(len(train_x))
    #print(len(val_x))
    helper = Helper()

    # tf idf
    tf_idf = TfidfVectorizer(sublinear_tf=True, max_df = 0.75, min_df=5, norm='l2',encoding='latin-1', ngram_range=(1, 2),stop_words='english')
    train_x_idf = tf_idf.fit_transform(train_x)
    test_x_idf = tf_idf.transform(test_x)
    X = tf_idf.transform(X)
    #val_x = tf_idf.transform(val_x)
    print(train_x[1:2])

    # normalize
    train_x_normalize = normalize(train_x_idf)
    test_x_normalize = normalize(test_x_idf)
    X = normalize(X)
    #val_x = normalize(val_x)
    # train_x_normalize, test_x_normalize = helper.run_pca(train_x_normalize,test_x_normalize)
    # Two features with highest chi-squared statistics are selected
    # SVM
    #train_x_normalize, test_x_normalize, train_y, test_y = train_test_split(X,y, train_size=0.8,
    #                                                                test_size=0.2)

    # logistic regression
    lr = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial')
    clf1 = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial')
    clf2 = MultinomialNB()
    clf3 = SGDClassifier(loss='log', max_iter=1000, tol=1e-3)
    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                          use_probas=True,
                          average_probas=False,
                          meta_classifier=lr)
    for clf, label in zip([clf1, clf2, clf3, sclf], 
                      ['LR', 
                       'MLNB',
                       'svm',
                       'StackingClassifier']):
        '''clf.fit(train_x_normalize, train_y)
        clf_pred = clf.predict(test_x_normalize)
        print(metrics.accuracy_score(test_y, clf_pred))
        print(metrics.classification_report(test_y, clf_pred))'''
        scores = model_selection.cross_val_score(clf, X, y, cv=3, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
            % (scores.mean(), scores.std(), label))

    '''clf.fit(train_x_normalize, train_y)

    # predict
    clf_pred = clf.predict(test_x_normalize)
    #val_pred = clf.predict(val_x)
    
    #write to validation file
    #df = pd.DataFrame(val_pred)
    #df.to_csv('validation.csv', index_label=['Id','Category'])
    #df.drop('0', axis=1)

    # evaluation on testt set
    print(metrics.accuracy_score(test_y, clf_pred))
    print(metrics.classification_report(test_y, clf_pred))'''





