import gc
import numpy as np

def cleanup(*args):
    for var in args:
        del var
        gc.collect()


def split_to_batches(data, batch):
    for i in range(0, len(data), batch):
        yield data[i:i + batch]


def generate_batch(X, y, batch_size, vocab_size, max_len, idx2word):
    while True:
        for j in range(0, len(X), batch_size):
            encoder_input_data = np.zeros((batch_size, max_len), dtype='float32')
            decoder_input_data = np.zeros((batch_size, max_len), dtype='float32')
            decoder_target_data = np.zeros((batch_size, max_len, vocab_size), dtype='float32')
            for i, (input_text, target_text) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):
                for t, word in enumerate(input_text.split()):
                    encoder_input_data[i, t] = idx2word[word]
                for t, word in enumerate(target_text.split()):
                    if t < len(target_text.split())-1:
                        decoder_input_data[i, t] = idx2word[word]
                    if t > 0:
                        decoder_target_data[i, t - 1, idx2word[word]] = 1.
            yield [encoder_input_data, decoder_input_data], decoder_target_data
