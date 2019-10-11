import numpy as np
import pandas as pd
import re


class DataClean:
    def __init__(self, filePath):
        self.filePath = filePath

    def read(self):
        # Read data from file 'filename.csv'
        # (in the same directory that your python process is based)
        # Control delimiters, rows, column names with read_csv (see later)
        self.data = pd.read_csv(self.filePath)

    def partition(self):
        # Takes the comments and parses them into an array of words
        partitionedComments = []
        for comment in self.data['comments']:
            partitionedComments.append(re.split('\W', comment))

            # filters out all redundant words in all comments
        uniqueWords = dict({})
        index = 0
        for comment in partitionedComments:
            for word in comment:
                if not word.lower() in uniqueWords.keys():
                    uniqueWords[word.lower()] = index
                    index += 1

        self.partitionedComments = partitionedComments
        self.uniqueWords = uniqueWords

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

    def buildYTrain(self, whatAreWeTesting):
        # allows us to switch from int to subreddit
        IntToSubReddit = dict({})
        index = 0
        for subReddit in self.data['subreddits']:
            if not subReddit in IntToSubReddit.values():
                IntToSubReddit[index] = subReddit
                index += 1

        self.IntToSubReddit = IntToSubReddit

        # given input of an integer we output a vector that determines whether its corresponding subreddit is expected
        subReddit = self.IntToSubReddit[whatAreWeTesting]

        yTrain = np.zeros(len(self.data['subreddits']))
        for index in range(len(self.data['subreddits'])):
            if self.data['subreddits'][index] == subReddit:
                yTrain[index] = 1
            else:
                yTrain[index] = 0
        self.yTrain = yTrain
        print(yTrain)


