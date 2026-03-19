from w2v.loader import Loader
from w2v.preprocess import preprocess, prepare_training
from w2v.model import Word2Vec

import nltk
from nltk.corpus import gutenberg
nltk.download('gutenburg')


# WINDOW_SIZE = 2


# preprocessed_source = preprocess(dataset_source)
# vocab, vocab_size, x_train, y_train = prepare_training(WINDOW_SIZE, preprocessed_source)

# model = Word2Vec(vocab, vocab_size, x_train, y_train)
# model.train(epochs=1000)


# print(f'Vocabulary: {model.vocabulary}')
# print(f'Window size: {model.window_size}')

# print(f'Predictions for word \'around\': {model.predict(word="around", predictions=3)}')
# print(f'Predictions for word \'around\': {model.predict(word="revolves", predictions=5)}')