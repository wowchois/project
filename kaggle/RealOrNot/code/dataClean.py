# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:42:14 2020

@author: insun
"""

import re,string
from nltk.tokenize import word_tokenize
from RealOrNot.code import engAbbrCorpus

abbreviations = engAbbrCorpus.abbreviations
contractions = engAbbrCorpus.contractions

class DataClean :
    def __init__(self, dataframe) :
        self.df = self.call_fn(dataframe)
        
    def call_fn(self,dataframe) :
        dataframe['c_text'] = dataframe['text'].apply(self.clean_text)
        dataframe['c_text'] = dataframe['c_text'].apply(self.remove_punct)
        dataframe['c_text'] = dataframe['c_text'].apply(self.remove_emoji)
        
        dataframe['c_text'] = dataframe['c_text'].apply(self.convert_abbrev_in_text)
        dataframe['c_text'] = dataframe['c_text'].apply(self.remove_contractions)

        return dataframe['c_text']
    
    def clean_text(self,text) :
        text = re.sub(r'https?://\S+', '', text) # Remove link
        text = re.sub(r'\n',' ', text) # Remove line breaks
        text = re.sub('\s+', ' ', text).strip() # Remove leading, trailing, and extra spaces
        return text
    
    def remove_punct(self,text):
        table = str.maketrans('','',string.punctuation)
        return text.translate(table)
    
    def remove_emoji(self,text):
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags = re.UNICODE)
        return emoji_pattern.sub(r'', text)
    
    def convert_abbrev(self,word):
        return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word

    def convert_abbrev_in_text(self,text): #약어를 원래 형태로 변형
        tokens = word_tokenize(text)
        tokens = [self.convert_abbrev(word) for word in tokens]
        text = ' '.join(tokens)
        return text    
    
    def remove_contractions(self,text): #수축 단어 원래 형태로 변형
        return contractions[text.lower()] if text.lower() in contractions.keys() else text



