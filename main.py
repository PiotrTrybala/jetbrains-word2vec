import numpy as np
import string
import nltk
from nltk.corpus import gutenberg, stopwords
from tqdm import tqdm

nltk.download('punkt')
nltk.download('gutenberg')

class Word2Vec:
    def __init__(self, embedding_dim=100, learning_rate=0.01, epochs=10, window_size=2):
        
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.window_size = window_size
        self.word_index = {}
        self.index_word = {}
        self.vocabulary_count = 0
        
        self.w1 = None # input -> hidden
        self.w2 = None # hidden -> output
    
    def preprocess(self, corpus: str) -> list[list[str]]:
        stop_words = set(stopwords.words('english'))
        sentences = corpus.split('.')
        cleaned_sentences = []
        
        for sentence in sentences:
            
            words = sentence.split()
            
            x = [word.strip(string.punctuation).lower() for word in words]
            x = [word for word in x if word and word not in stop_words]
            
            if len(x) > 1:
                cleaned_sentences.append(x)
                
        return cleaned_sentences
    
    def prepare_vocabulary(self, sentences):
        
        vocabulary = set()
        for sentence in sentences:
            for word in sentence:
                vocabulary.add(word)
        
        sorted_vocabulary = sorted(list(vocabulary))
        self.vocabulary_count = len(sorted_vocabulary)
        self.word_index = {word: i for i, word in enumerate(sorted_vocabulary)}
        self.index_word = {i: word for i, word in enumerate(sorted_vocabulary)}
        
        self.w1 = np.random.uniform(-0.1, 0.1, 
                                    (self.vocabulary_count, self.embedding_dim))
        self.w2 = np.random.uniform(-0.1, 0.1,
                                    (self.embedding_dim, self.vocabulary_count))
    
    def _prepare_training_data(self, sentences):
        data = []
        
        for sentence in sentences:
            for i, word in enumerate(sentence):
                
                target = self.word_index[word]
                start = max(0, i - self.window_size)
                end = min(len(sentence), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        context = self.word_index[sentence[j]]
                        data.append((target, context))
            
        return data
    
    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def train(self, corpus):
        sentences = self.preprocess(corpus)
        self.prepare_vocabulary(sentences)
        traning_pairs = self._prepare_training_data(sentences)
        
        print(f'vocabulary size: {self.vocabulary_count}')
        print(f'traning samples: {len(traning_pairs)}')
        
        for epoch in range(self.epochs):
            loss = 0
            
            for target_idx, context_idx in tqdm(traning_pairs, f'Epoch {epoch}'):
                
                hidden = self.w1[target_idx]
                output = np.dot(hidden, self.w2)
                prediction = self._softmax(output)
                
                error = prediction
                error[context_idx] -= 1
                
                dw2 = np.outer(hidden, error)
                dw1 = np.dot(self.w2, error)
                
                self.w1[target_idx] -= self.learning_rate * dw1
                self.w2 -= self.learning_rate * dw2
                
                p_correct = np.clip(prediction[context_idx], 1e-12, 1.0)
                
                loss += -np.log(p_correct)
                
            print(f'Loss: {loss / len(traning_pairs):.4f}')
    
    def predict(self, word, predictions=5):
        
        if word not in self.word_index: return []
        
        idx = self.word_index[word]
        hidden = self.w1[idx]
        output = np.dot(hidden, self.w2)
        prediction = self._softmax(output)
        
        top_indices = np.argsort(prediction)[-predictions:][::-1]
        return [(self.index_word[i], prediction[i]) for i in top_indices]
    
    # def get_embedding(self, word):
    #     pass
    
    # def cosine_similarity(self, word1, word2):
    #     pass
    
raw_text = gutenberg.raw('carroll-alice.txt')

model = Word2Vec(embedding_dim=50, learning_rate=0.01, epochs=5, window_size=2)
model.train(raw_text)

print("Context words for 'alice' are:", model.predict('alice'))