import string
import unicodedata
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import el_core_news_md

nlp = el_core_news_md.load()


def clean_text(text_list):
    """Takes a list of greek sequences, converts to lowercase and strips punctuation and notation"""
    text_ = []
    d = {ord('\N{COMBINING ACUTE ACCENT}'): None}
    for text in text_list:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = unicodedata.normalize('NFD', text).translate(d)
        text_.append(text)
    return text_


def tag_start_end_sentences(decoder_input_sentence):
    """Takes the targets list and tags start <BOS> and end <EOS>"""
    bos = "<BOS>"
    eos = "<EOS>"
    final_target = [bos + text + eos for text in decoder_input_sentence]
    return final_target


def vocab_creator(text_lists, vocab_size):
    """Takes a list with all sequences, creates vocabulary list and two dictionaries after tokenization, word to index
       and index to word"""
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(text_lists)
    dictionary = tokenizer.word_index

    word2idx = {}
    idx2word = {}
    vocab = []

    for k, v in dictionary.items():
        if v < vocab_size:
            word2idx[k] = v
            idx2word[v] = k
            vocab.append(k)

        if v >= vocab_size - 1:
            continue

    return word2idx, idx2word, vocab


def text2seq(encoder_text, decoder_text, vocab_size):
    tokenizer = Tokenizer(num_words=vocab_size, filters='!\"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n', lower=False)
    tokenizer.fit_on_texts(encoder_text)
    tokenizer.fit_on_texts(decoder_text)
    encoder_sequences = tokenizer.texts_to_sequences(encoder_text)
    decoder_sequences = tokenizer.texts_to_sequences(decoder_text)

    return encoder_sequences, decoder_sequences


def padding(encoder_sequences, decoder_sequences, max_len):
    encoder_input_data = pad_sequences(encoder_sequences, maxlen=max_len, dtype='int32', padding='post', truncating='post')
    decoder_input_data = pad_sequences(decoder_sequences, maxlen=max_len, dtype='int32', padding='post', truncating='post')

    return encoder_input_data, decoder_input_data


def create_embedding_matrix(vocab, vocab_size, embedding_dim):
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for i, word in enumerate(vocab):
        embedding_matrix[i] = nlp(word).vector
    return embedding_matrix


def create_decoder_output(decoder_input_data, num_samples, max_len, vocab_size):
    decoder_output_data = np.zeros((num_samples, max_len, vocab_size), dtype="uint8")

    for i, seqs in enumerate(decoder_input_data):
        for j, seq in enumerate(seqs):
            if j > 0:
                decoder_output_data[i][j][seq] = 1.

    return decoder_output_data

