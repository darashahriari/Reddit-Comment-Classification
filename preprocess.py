#!/usr/bin/env python3
import re
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords

class Preprocess:
    def __init__(self):
        self.stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER', 'URL'])

    def processText(self, list_of_data):
        processedText = []
        for text in list_of_data:
            processedText.append((self._processtext(text['comments']), text['subreddits']))
        return processedText

    def _processtext(self, data):
        data = data.lower()  # convert text to lower-case
        data = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', data)  # remove URLs
        data = re.sub('@[^\s]+', 'AT_USER', data)  # remove usernames
        data = re.sub(r'#([^\s]+)', r'\1', data)  # remove the # in #hashtag
        data = word_tokenize(data)  # remove repeated characters (helloooooooo into hello)
        return [word for word in data if word not in self._stopwords]
