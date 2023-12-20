#%%
import numpy as np
import re

text = '''
So do all who live to see such times. But that is not for them to decide. \
All we have to decide is what to do with the time that is given us'''

#%%
#Tokenize the text
def tokenize(text):
    #The problem is .split() is it includes punctuation
    #Using the regex expresion
    #The regex exp defines a search pattern in a string
    #Learn more abour regex in regex101
    pat = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pat.findall(text)

    
#%%
def generate_dict_data(tokens):

    word_to_index = {}
    index_to_word = {}
    count = 0

    for token in tokens:
        #convert word to lowercase
        word = token.lower()
        #In case there are repeated words 
        if word_to_index.get(word) == None:
            word_to_index.update({word : count})
            index_to_word.update({count : token})
        count += 1
    return word_to_index, index_to_word

#%%
#Define the one hot encoding for each word in the vocab


