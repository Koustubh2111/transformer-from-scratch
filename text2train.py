#%%
import numpy as np
import re

text = '''
So do all who live to see such times. But that is not for them to decide. \
All we have to decide is what to do with the time that is given us'''

#Tokenize the text
def tokenize(text):
    #The problem is .split() is it includes punctuation
    #Using the regex expresion
    #The regex exp defines a search pattern in a string
    #Learn more abour regex in regex101
    pat = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pat.findall(text)

    
def generate_dict_data(tokens):

    word_to_index = {}
    index_to_word = {}
    count = 0
    #Separate variable for the word corpus for unique words
    corpus = []

    for token in tokens:
        #convert word to lowercase
        word = token.lower()
        corpus.append(word) #Corpus will contain repeated words in lowercase and not diff from tokens
        #In case there are repeated words 
        if word_to_index.get(word) == None:
            word_to_index.update({word : count})
            index_to_word.update({count : word})
        count += 1
    return corpus, word_to_index, index_to_word


#Define the one hot encoding for each word in the vocab
def one_hot_encode(n_corpus, word, word_to_index):

    one_hot_word = np.zeros(n_corpus)
    one_hot_word[word_to_index.get(word)] = 1
    return one_hot_word

#Define the training data 
#The skip gram method is implemented here where each input word has two context words 
def generate_training_data(corpus, word_to_index, index_to_word):
    training_data = []
    len_corpus = len(corpus)
    half_window = 4
    for i in range(len_corpus):
        X = one_hot_encode(len_corpus, corpus[i], word_to_index)

        #To get the output we need to look the adjacent 4 words in the corpus (lowercase tokens)
        #for corpus[i], four outputs are present corpus[i-2], corpus[i-1], corpus[i+1] & corpus[i+2]
        #Have to deal with edge cases
        #JOB : Working fine but implement a dynamic sliding window
         

        if i == 0:
            context_words  = [corpus[idx] for idx in range(i+1, half_window + 1)]
        
        elif i == len_corpus - 1:
            context_words = [corpus[idx] for idx in range(len_corpus - 2 ,len_corpus - 2 - half_window  , -1 )]

        else:
            context_words = [corpus[idx] for idx in range(i-1, i - 1- half_window, -1) if idx >=0]
            context_words.extend([corpus[idx] for idx in range(i + 1, i + 1 + half_window) if idx < len_corpus])

        #After generating a context word list, append the one hot encode of each context word of each word as a separate input to training data
        
        for context_word in context_words:
            y = one_hot_encode(len_corpus, context_word, word_to_index)
            training_data.append([X,y])

    return training_data


