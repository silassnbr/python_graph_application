
from gensim.models import KeyedVectors
glove = KeyedVectors.load_word2vec_format('glove.42B.300d.txt', binary=False,no_header=True)
glove.save('model.bin')