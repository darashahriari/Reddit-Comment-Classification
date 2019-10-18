import os
import numpy as np
import pandas as pd
import re
import nltk
import time
from sklearn.datasets import load_files
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

class MultinomialBayes :
    def __init__(self, filePath):
        self.filePath = filePath

    def read(self):
        print('\n \n \n Reading File ...')

        data = pd.read_csv(self.filePath)

        print('Cleaning Text ...')

        self.y = data['subreddits']
        X = data['comments']
        stemmer = WordNetLemmatizer()
        documents = []
        for sen in range(0, len(X)):
            # Remove all the special characters
            document = re.sub(r'\W', ' ', str(X[sen]))
            # remove all single characters
            document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
            # Remove single characters from the start
            document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
            # Substituting multiple spaces with single space
            document = re.sub(r'\s+', ' ', document, flags=re.I)
            # Removing prefixed 'b'
            document = re.sub(r'^b\s+', '', document)
            # Converting to Lowercase
            document = document.lower()
            # Lemmatization
            document = document.split()
            document = [stemmer.lemmatize(word) for word in document]
            document = ' '.join(document)
            documents.append(document)
            
        self.X = documents

    def featureSelect(self):

        print('Selecting Features ...')
        
        self.tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.03, stop_words='english')

        X = self.tfidf_vectorizer.fit_transform(self.X)

        self.xtrain = X
        self.ytrain = self.y

        print('Feature Selection Complete:')
        print('Starting Taining ...\n \n')
        
        self.text_clf = Pipeline([('tfidf', TfidfTransformer()),('clf', MultinomialNB(alpha=.83,fit_prior=True, class_prior=[.05,.05,.05,.05,.05,.05,.05,.05,.05,.05,.05,.05,.05,.05,.05,.05,.05,.05,.05,.05])),]) 
        

    def fit(self):
        self.text_clf.fit(self.xtrain, self.ytrain)

    def pred(self, filepath):
        testData = pd.read_csv(filepath)
        X = testData['comments']
        stemmer = WordNetLemmatizer()
        documents = []
        for sen in range(0, len(X)):
            # Remove all the special characters
            document = re.sub(r'\W', ' ', str(X[sen]))
            # remove all single characters
            document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
            # Remove single characters from the start
            document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
            # Substituting multiple spaces with single space
            document = re.sub(r'\s+', ' ', document, flags=re.I)
            # Removing prefixed 'b'
            document = re.sub(r'^b\s+', '', document)
            # Converting to Lowercase
            document = document.lower()
            # Lemmatization
            document = document.split()
            document = [stemmer.lemmatize(word) for word in document]
            document = ' '.join(document)
            documents.append(document)
        
        X = self.tfidf_vectorizer.transform(documents)

        self.xtest = X

        predictions = self.text_clf.predict(self.xtest)
        df = pd.DataFrame({'Category': predictions})
        df.to_csv(index=True, path_or_buf='ans.csv')

    def fitTest(self):
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.xtrain, self.ytrain, test_size=0.2, random_state=0)
        start_time = time.time()
        self.text_clf.fit(self.xtrain, self.ytrain) 
        print("--- %s runtime in seconds ---" % (time.time() - start_time))

    def predTest(self):
        self.ypred =  self.text_clf.predict(self.xtest)
        print(classification_report(self.ytest,self.ypred))
        print(accuracy_score(self.ytest, self.ypred))
        
cleaner = MultinomialBayes("data/reddit_train.csv")
cleaner.read()
cleaner.featureSelect()
cleaner.fitTest()
cleaner.predTest()
#cleaner.fit()
#cleaner.pred("data/reddit_test 2.csv")
