#!/usr/bin/env python3
import numpy as np
import numpy.linalg
import numpy.random
import pandas as pd
from NaiveBayes import NaiveBayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from mlxtend.classifier import StackingClassifier, StackingCVClassifier
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
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    # import data
    data = pd.read_csv('reddit_train.csv')
    val = pd.read_csv('reddit_test.csv')
    
    # split data into train and test sets
    val_x = val['comments']
    X = data['comments']
    y = data['subreddits']
    helper = Helper()

    # tf idf
    tf_idf = TfidfVectorizer(sublinear_tf=True, max_df = 0.03, min_df=2, norm='l2',encoding='latin-1', ngram_range=(1, 2),stop_words='english')
    X = tf_idf.fit_transform(X)
    val_x = tf_idf.transform(val_x)

    # normalize
    X = normalize(X)
    val_x = normalize(val_x)
    
    # Two features with highest chi-squared statistics are selected
    chi2_features = SelectKBest(chi2, k = 15500)
    X_kbest_features = chi2_features.fit_transform(X,y)
    val_kbest_features = chi2_features.transform(val_x)
    print('Original feature number:', X.shape[1])
    print('Reduced feature number:', X_kbest_features.shape[1])
    train_x_normalize, test_x_normalize, train_y, test_y = train_test_split(X_kbest_features,y, train_size=0.8,
                                                                    test_size=0.2)    

    #initialize classifiers
    clf1 = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial')
    clf2 = MultinomialNB()
    clf3 = SGDClassifier(loss='modified_huber', max_iter=1000, tol=1e-3)
    sclf = StackingClassifier(classifiers=[clf2, clf3],
                          use_probas=True,
                          average_probas=False,
                          meta_classifier=clf1)
    for clf, label in zip([clf2, clf3, sclf], 
                      [
                       'nb',
                       'svm',
                       'StackingClassifier']):
        scores = model_selection.cross_val_score(clf,X_kbest_features , y, cv=5, scoring='accuracy')
        print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), label))
    
    sclf.fit(train_x_normalize, train_y)
    val_pred = sclf.predict(val_kbest_features)

    #write to validation file
    df = pd.DataFrame({'Category': val_pred})
    df.to_csv(index=True, path_or_buf='validation_stacking.csv', index_label='Id')




