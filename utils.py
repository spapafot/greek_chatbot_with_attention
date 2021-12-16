import gc
import numpy as np
import el_core_news_md

nlp = el_core_news_md.load()


def cleanup(*args):
    for var in args:
        del var
        gc.collect()


def split_to_batches(data, batch):
    for i in range(0, len(data), batch):
        yield data[i:i + batch]


def create_embedding_matrix(vocab, vocab_size, embedding_dim):
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for i, word in enumerate(vocab):
        embedding_matrix[i] = nlp(word).vector
    return embedding_matrix
