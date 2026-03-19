from w2v.loader import Loader
from w2v.preprocess import preprocess, prepare_training
from w2v.model import Word2Vec

# 1. Load dataset used for learning

dataset_source = Loader.load_zip("./datasets/Gutenburg.zip")

# 2. Preprocess and prepare training data

preprocessed_source = preprocess(dataset_source)
vocab, vocab_size, x_train, y_train = prepare_training(preprocessed_source)

# 3.1. Train model
model = Word2Vec(vocab, vocab_size, x_train, y_train)
model.train(epochs=1000)

# 3.1. Display internal variables

print(f'Vocabulary: {model.vocabulary}')
print(f'Window size: {model.window_size}')

#. 4. Use model to predict

print(f'Predictions for word \'around\': {model.predict(word="around", predictions=3)}')
print(f'Predictions for word \'around\': {model.predict(word="revolves", predictions=5)}')