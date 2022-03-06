import gc
import numpy as np
import el_core_news_md

nlp = el_core_news_md.load()


def cleanup(*args):
    """clean unused vars, did not have a memory error so far so did not use"""
    for var in args:
        del var
        gc.collect()


def split_to_batches(data, batch):
    """manual batch splitting just in case"""
    for i in range(0, len(data), batch):
        yield data[i:i + batch]


def create_embedding_matrix(vocab, vocab_size, embedding_dim):
    """create embedding matrix with the greek pretrained SPACY greek model vectors"""
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for i, word in enumerate(vocab):
        embedding_matrix[i] = nlp(word).vector
    return embedding_matrix


def get_vocab_size(data):
    """takes a list of clean strings and returns unique number of tokens"""
    return len(set(" ".join(data).split()))

