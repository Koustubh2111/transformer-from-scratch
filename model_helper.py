import numpy as np
import text2train

text = '''
So do all who live to see such times. But that is not for them to decide. \
All we have to decide is what to do with the time that is given us'''

initialize  = { \
    'text' : text, \
    'half_window' : 4, \
    'n_epochs' : 50, \
    'learning_rate' : 0.01 \
}

class word2vec():
    
    def __init__(self, initialize_dict):
        self.text = initialize_dict['text']
        self.half_window = initialize_dict['half_window']
        self.n_epochs = initialize_dict['n_epochs']
        self.learning_rate = initialize_dict['learning_rate']
        pass

    def generate_training_data(self):

        t2t = text2train(self.text, self.half_window)
        #First tokemize the data and store tokens
        t2t.tokenize()
        #Generate dict data and store
        t2t.generate_dict_data()

        #Store dict data in this class
        #JOB : Use inheritance for cleaner code
        #Note both corpus and word_2_index are available after running gen_dict_data
        self.corpus = t2t.corpus
        self.word_to_index = t2t.word_to_index

        #Get training data
        self.training_data = t2t.generate_training_data()
        pass

    def softmax(self,x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def forward_pass(self, x):
        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y_c = self.softmax(u)
        return y_c, h, u
    
    
    








