import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

"""
preprocess - split sentences into words and remove all the stopwords like the, is, in etc.
and store them into lists
"""

def preprocess(corpus: str) -> list[str]:
    
    training_data: list[str] = []
    stop_words = set(stopwords.words('english'))
    
    sentences = corpus.split('.')
    
    for i in range(len(sentences)):
        
        sentences[i] = sentences[i].strip()
        sentence = sentences[i].split()
        
        x = [word.strip(string.punctuation) for word in sentence if word not in stop_words]
        x = [word.lower() for word in x]
        training_data.append(x)
    
    return training_data

"""
prepare_training - used for preparing two lists: one is target word, and the other is for
all context that applies to it

"""

def prepare_training(window_size: int, sentences: list[list[str]]):
    data = {}
    
    for sentence in sentences:
        for word in sentence:
            if word not in data:
                data[word] = 1
            else:
                data[word] += 1
    
    vocabulary_size = len(data)
    vocabulary = {}
    
    for i in range(len(data)):
        vocabulary[data[i]] = i
        
    x_train, y_train = [], []
    for sentence in sentences:
        for i in range(len(sentence)):
            center_word = [0 for _ in range(vocabulary_size)]
            center_word[vocabulary[sentence[i]]] = i
            context = [0 for _ in range(vocabulary_size)]
            
            for j in range(i - window_size, i + window_size):
                if i != j and j >= 0 and j < len(sentences):
                    context[vocabulary[sentences[j]]] += 1
                    x_train.append(center_word)
                    y_train.append(context)

    return vocabulary, vocabulary_size, x_train, y_train