from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


def create_embedding_layer(vocab_size, embedding_dim, max_len, embedding_matrix=0):
    embedding_layer = layers.Embedding(input_dim=vocab_size,
                                       output_dim=embedding_dim,
                                       input_length=max_len,
                                       weights=[embedding_matrix],
                                       mask_zero=True,
                                       trainable=False)
    return embedding_layer


def create_seq2seq_model(embedding_dim, vocab_size, max_len, embedding_layer):
    encoder_inputs = layers.Input(shape=(max_len,), dtype='int32')
    encoder_embedding = embedding_layer(encoder_inputs)
    encoder_LSTM = layers.LSTM(embedding_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)

    decoder_inputs = layers.Input(shape=(max_len,), dtype='int32')
    decoder_embedding = embedding_layer(decoder_inputs)
    decoder_LSTM = layers.LSTM(embedding_dim, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])
    outputs = layers.Dense(vocab_size, activation='softmax')(decoder_outputs)

    model = models.Model([encoder_inputs, decoder_inputs], outputs)

    return model


def run_inference(model, embedding_layer):

    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[3].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = models.Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]  # input_2
    decoder_state_input_h = layers.Input(shape=(300,))
    decoder_state_input_c = layers.Input(shape=(300,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[4]
    decoder_embedding = embedding_layer(decoder_inputs)
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[5]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = models.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model


def decode_sequence(input_seq, encoder_model, decoder_model, word2idx, idx2word):

    input_seq = format_to_padded_tokens(input_seq, word2idx, max_len=25)
    states_value = encoder_model.predict([input_seq])

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx['<BOS>']
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = idx2word[sampled_token_index + 1]
        decoded_sentence += ' ' + sampled_token

        if sampled_token == '<EOS>' or len(decoded_sentence) > 50:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence


def format_to_padded_tokens(sentence, word2idx, max_len):

    formatted_sentence = []
    for i in sentence.split():
        formatted_sentence.append(word2idx[i])

    return pad_sequences([formatted_sentence], maxlen=max_len, padding="post", truncating="post")

