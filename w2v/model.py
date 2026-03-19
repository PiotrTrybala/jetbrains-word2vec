import numpy as np
class Word2Vec(object):
    
    def __init__(self):
        
        self.hidden_layer_size = 10
        self.x_train = []
        self.y_train = []
        self.window_size = 2
        self.alpha = 0.001
        self.words = []
        self.word_index = {}
    
    """"""
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    """"""
    def init(self, vocabulary_size, vocabulary):
        
        self.vocabulary_size = vocabulary_size
        self.input_to_hidden_weights = np.random.uniform(-0.8, 0.8, 
                                                         (self.vocabulary_size, self.hidden_layer_size))
        self.hidden_to_output_weights = np.random.uniform(-0.8, 0.8,
                                                          (self.hidden_layer_size, self.vocabulary_size))

        self.words = vocabulary
        for i in range(len(vocabulary)):
            self.word_index[vocabulary[i]] = i
    
    """"""
    def feed_forward(self, input_vector):
        self.hidden = np.dot(self.input_to_hidden_weights.T, 
                             input_vector).reshape(self.hidden_layer_size, 1)
        
        self.output = np.dot(self.hidden_to_output_weights.T, 
                             self.hidden)
        
        self.activations = self.softmax(self.output)
    
    """"""
    def propagate_backward(self, input_vector, output_vector):
        
        error = self.activations - np.asarray(output_vector).reshape(self.vocabulary_size, 1)
        
        hidden_layer_error = np.dot(self.hidden, error.T) 
        
        X = np.array(input_vector).reshape(self.vocabulary_size, 1)
        input_layer_error = np.dot(X, np.dot(self.hidden_to_output_weights, error).T)
         
        
        self.hidden_to_output_weights = self.hidden_to_output_weights - self.alpha * hidden_layer_error
        self.input_to_hidden_weights = self.input_to_hidden_weights - self.alpha * input_layer_error
        
    
    """"""
    def train(self, epochs):
        
        for epoch in range(1, epochs):
            self.loss = 0
            
            for j in range(len(self.x_train)):
                self.feed_forward(self.x_train[j])
                self.propagate_backward(self.x_train[j], self.y_train[j])
                
                cost = 0
                for word in range(self.vocabulary_size):
                    if self.y_train[j][word]:
                        self.loss -= -1 * self.output[word]
                        cost += 1
                self.loss += cost * np.log(np.sum(np.exp(self.output)))
            
            print(f'Epoch #{epoch}, loss = {self.loss}')
            self.alpha *= 1 / (1 + self.alpha * epoch)
            
    """"""
    def predict(self, word, predictions):
        
        if word in self.words:
            index = self.word_index[word]
            input_vector = [0 for i in range(self.vocabulary_size)]
            input_vector[index] = 1
            
            prediction = self.feed_forward(input_vector)
            
            print(f'Prediction vector for word {word}: {prediction}')
            
            output = {}
            for i in range(self.vocabulary_size):
                output[prediction[i][0]] = i
            
            
            top_context_words = []
            for k in sorted(output,reverse=True):
                top_context_words.append(self.words[output[k]])
                if(len(top_context_words)>=predictions):
                    break
            
            return top_context_words
        else:
            print(f'Word {word} is not present in vocabulary set')
