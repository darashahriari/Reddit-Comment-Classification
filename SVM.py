#!/usr/bin/env python3
import numpy as np
import numpy.linalg
import numpy.random
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
#from DataClean import DataClean
from NaiveBayes import NaiveBayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn import metrics
from helper import Helper
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2 

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
    helper = Helper()

    '''vectorizer = CountVectorizer()
    training_x = vectorizer.fit_transform(train_x)
    testing_x = vectorizer.transform(test_x)
    print(testing_x.shape)
    print(training_x)'''

    # tf idf
    tf_idf = TfidfVectorizer(sublinear_tf=True, max_df = 0.75, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
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
    #train_x_normalize, test_x_normalize = helper.run_pca(train_x_normalize,test_x_normalize)
    # Two features with highest chi-squared statistics are selected 
    chi2_features = SelectKBest(chi2, k = 15000) 
    X_kbest_features = chi2_features.fit_transform(X,y)
    print('Original feature number:', train_x_normalize.shape[1]) 
    print('Reduced feature number:', X_kbest_features.shape[1])
    # SVM
    #train_x_normalize, test_x_normalize, train_y, test_y = train_test_split(X,y, train_size=0.8,
     #                                                               test_size=0.2)
    #clf = LinearSVC(random_state=0, tol=1e-5)
    clf =SGDClassifier(max_iter=1000, tol=1e-3) 
    clf.fit(train_x_normalize, train_y)

    # predict
    clf_pred = clf.predict(test_x_normalize)
    #val_pred = clf.predict(val_x)

    #write to validation file
    #df = pd.DataFrame(val_pred)
    #df.to_csv('validation.csv', index_label=['Id','Category'])

    # evaluation on testt set
    print(metrics.accuracy_score(test_y, clf_pred))
    print(metrics.classification_report(test_y, clf_pred))

