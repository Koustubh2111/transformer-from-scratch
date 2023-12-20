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
#Define a numeric mapping for each token
def mapping(tokens):
    word_to_num = {}
    for num, token in enumerate(set(tokens)):
        word_to_num[token] = num
    return word_to_num


# %%
