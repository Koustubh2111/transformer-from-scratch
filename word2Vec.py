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
np.random.seed(42)
word_to_num = mapping(tokens)

def generate_training_data(tokens, word_to_num, window):
    X = []
    y = []
    n_tokens = len(tokens)
    
    for i in range(n_tokens):
        idx = concat(
            range(max(0, i - window), i), 
            range(i, min(n_tokens, i + window + 1))
        )
        for j in idx:
            if i == j:
                continue
            X.append(one_hot_encode(word_to_num[tokens[i]], len(word_to_num)))
            y.append(one_hot_encode(word_to_num[tokens[j]], len(word_to_num)))
    
    return np.asarray(X), np.asarray(y)

def concat(*iterables):
    for iterable in iterables:
        yield from iterable

def one_hot_encode(id, vocab_size):
    res = [0] * vocab_size
    res[id] = 1
    return res

# %%
