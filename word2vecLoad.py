from gensim.models import KeyedVectors
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
model.save('vord2vec.bin')