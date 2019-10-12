import numpy as np
import pandas as pd
import re
import os
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import csv

class DataClean:
    def __init__(self, filePath):
        self.filePath = filePath

    def read(self):
        # Read data from file 'filename.csv'
        # (in the same directory that your python process is based)
        # Control delimiters, rows, column names with read_csv (see later)
        self.data = pd.read_csv(self.filePath)
        print(self.data)
    
    def partition(self):
        #Takes the comments and parses them into an array of words
        partitionedComments = pd.read_csv('partition.csv',error_bad_lines=False)
        '''for comment in self.data['comments']:
            partitionedComments.append(re.split('\W',comment)) 
        
        #filters out all redundant words in all comments
        # Load stop words'''
        stop_words = stopwords.words('english')
        uniqueWords = dict({})
        index = 0
        for comment in partitionedComments:
            for word in comment:
                if word not in stop_words and not '':
                    if not word.lower() in uniqueWords.keys():
                        uniqueWords[word.lower()] = index
                        index += 1

        self.partitionedComments = partitionedComments
        self.uniqueWords = uniqueWords
        #a = np.asarray(self.partitionedComments)
        #np.savetxt("partition.csv", a, fmt='%s', delimiter=",")
        print('done partition')
        with open('unique_words.csv', 'wb') as f:  # Just use 'w' mode in 3.x
                w = csv.DictWriter(f, self.uniqueWords.keys())
                w.writeheader()
                w.writerow(self.uniqueWords)
        print('done unique')
        
    def buildXTrain(self):
        # initialize the x vectors
        examples = []
        for comment in self.partitionedComments:
            binaryFeatures = np.zeros(len(self.uniqueWords))
            for word in comment:
                if word.lower() in self.uniqueWords.keys():
                    binaryFeatures[self.uniqueWords[word.lower()]] = 1
            examples.append(binaryFeatures)
        self.xTrain = np.asarray(examples)
        df = pd.DataFrame(self.xTrain)
        df.to_csv(index=False)

    def buildYTrain(self, whatAreWeTesting):
        # allows us to switch from int to subreddit
        IntToSubReddit = dict({})
        index = 0
        for subReddit in self.data['subreddits']:
            if not subReddit in IntToSubReddit.values():
                    IntToSubReddit[index] = subReddit
                    index += 1

        self.IntToSubReddit = IntToSubReddit

        #given input of an integer we output a vector that determines whether its corresponding subreddit is expected
        subReddit = self.IntToSubReddit[whatAreWeTesting]

        yTrain = np.zeros(len(self.data['subreddits']))
        for index in range(len(self.data['subreddits'])):
            if self.data['subreddits'][index] == subReddit:
                yTrain[index] = 1
            else:
                yTrain[index] = 0
        self.yTrain = yTrain
        df = pd.DataFrame(self.yTrain)
        df.to_csv(index=False)

cleaner = DataClean("reddit_train.csv")
cleaner.read()
cleaner.partition()
cleaner.buildXTrain()
cleaner.buildYTrain(9)


