import numpy as np
import pandas as pd
import re
import nltk
from sklearn.datasets import load_files
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import string


class DataClean :
    def __init__(self, filePath):
        self.filePath = filePath
    
    

    def tokenization(self, text):
        text = re.split('\W', text)
        return text
    
    def remove_stopwords(self, text):
        stopword = nltk.corpus.stopwords.words('english')
        text = [word for word in text if word not in stopword]
        return text

    def stemming(self, text):
        ps = nltk.PorterStemmer()
        text = [ps.stem(word) for word in text]
        return text

    def lemmatizer(self, text):
        wn = nltk.WordNetLemmatizer()
        text = [wn.lemmatize(word) for word in text]
        return text


    def read(self):
        data = pd.read_csv(self.filePath)
        y = data['subreddits']
        Lemmatizer = WordNetLemmatizer()
        documents = []
        data['comments'] = data['comments'].apply(lambda x: self.tokenization(x.lower()))
        data['comments'] = data['comments'].apply(lambda x: self.remove_stopwords(x))
        data['comments'] = data['comments'].apply(lambda x: self.stemming(x))
        data['comments'] = data['comments'].apply(lambda x: self.lemmatizer(x))


        for comment in data['comments']:
            newLine = ""
            for word in comment:
                if(len(word) > 1):
                    newLine = newLine + ' '+ Lemmatizer.lemmatize(word.lower()) 
            
            documents.append(newLine)
        
        #print(documents)
        
        vectorizer = TfidfVectorizer(min_df=2, max_df=0.03, stop_words='english')
        X = vectorizer.fit_transform(documents).toarray()
        self.vocabulary = vectorizer.get_feature_names()
        self.xtrain = X
        self.ytrain = y
        
    def fit(self):
        self.classifier = MultinomialNB()
        self.classifier.fit(self.xtrain, self.ytrain) 

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
        
        vectorizer = TfidfVectorizer( vocabulary=self.vocabulary)
        X = vectorizer.fit_transform(documents).toarray()

        predictions = self.classifier.predict(X)
        df = pd.DataFrame({'Category': predictions})
        df.to_csv(index=True, path_or_buf='ans.csv')

    def fitTest(self):
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.xtrain, self.ytrain, test_size=0.2, random_state=0)
        #self.classifier = MultinomialNB()
        #self.classifier = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial')
        #self.classifier = LinearSVC(random_state=0, tol=1e-5)
        self.classifier = SGDClassifier(max_iter=1000, tol=1e-3)
        self.classifier.fit(self.xtrain, self.ytrain) 

    def predTest(self):
        self.ypred = self.classifier.predict(self.xtest)
        print(confusion_matrix(self.ytest,self.ypred))
        print(classification_report(self.ytest,self.ypred))
        print(accuracy_score(self.ytest, self.ypred))
        
cleaner = DataClean("reddit_train.csv")
cleaner.read()
print("done read")
cleaner.fitTest()
cleaner.predTest()
#cleaner.fit()
#cleaner.pred("/Users/dara/desktop/mini project 2/reddit_test 2.csv")
