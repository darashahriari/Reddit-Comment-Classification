#!/usr/bin/env python3
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn import metrics
from nltk.corpus import stopwords

if __name__ == '__main__':
    # import data
    data = pd.read_csv('/Users/Anna/COMP551-Project2/data/reddit_train.csv')
    # test_data = pd.read_csv('/Users/Anna/COMP551-Project2/data/reddit_test.csv')
    x = data['comments']
    y = data['subreddits']

    #tf_idf
    tf_idf = TfidfVectorizer(sublinear_tf=True, max_df=0.75, min_df=3, norm='l2', ngram_range=(1, 2), encoding='latin-1', stop_words='english')
    x = tf_idf.fit_transform(x)

    # normalize
    x = normalize(x)

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
    print(clf_pred)

    # evaluation on testt set
    print(metrics.accuracy_score(test_y, clf_pred))
    print(metrics.classification_report(test_y, clf_pred))
