from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

word_vector = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

print(len(word_vector))


def odd_one_out(words):
    all_word_vectors = [word_vector[w] for w in words]
    all_word_vectors = np.array(all_word_vectors)

    avg_vector = all_word_vectors.mean(axis=0)

    odd_one_out_word = None
    min_similarity = 1.0

    for w in words:
        sim = cosine_similarity([word_vector[w]], [avg_vector])
        if sim < min_similarity:
            min_similarity = sim
            odd_one_out_word = w

    return odd_one_out_word


list_name = ['apple', 'mango', 'juice', 'party', 'orange', 'grapes']

odd = odd_one_out(list_name)
print('Odd from', str(list_name), ' is ', odd)


def predict_word(a, b, c, word_vector):
    a, b, c = a.lower(), b.lower(), c.lower()
    max_similarity = -100
    d = None
    words = word_vector.vocab.keys()
    wa, wb, wc = word_vector[a], word_vector[b], word_vector[c]
    for w in words:
        if w in [a, b, c]:
            continue
        wv = word_vector[w]
        sim = cosine_similarity([wb - wa], [wv - wc])
        if sim > max_similarity:
            max_similarity = sim
            d = w
    return d


words = ('man', 'woman', 'king')
print(predict_word(*words, word_vector))


sim_word = word_vector.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(sim_word)
