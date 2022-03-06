from tensorflow.keras import models, layers
import tensorflow as tf
import tensorflow_addons as tfa


class Encoder(models.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size, embedding_matrix, max_length_input):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length_input, weights=[embedding_matrix], trainable=False)
        self.lstm_layer = layers.LSTM(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, h, c = self.lstm_layer(x, initial_state=hidden)
        return output, h, c

    def initialize_hidden_state(self):
        # in case of initialization memory error for larger projects use utilities cleanup()
        return [tf.zeros((self.batch_size, self.enc_units)), tf.zeros((self.batch_size, self.enc_units))]


class Decoder(models.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size, max_length_input, max_length_output, embedding_matrix, attention_type='bahdanau'):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.attention_type = attention_type
        self.max_length_output = max_length_output

        # do not use weights if you want to train with random weights, i am using spacy's pretrained vectors
        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length_input,weights=[embedding_matrix], trainable=False)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        self.attention_mechanism = self.build_attention_mechanism(self.dec_units, None, batch_size * [max_length_input], self.attention_type)
        self.rnn_cell = self.build_rnn_cell(batch_size)
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc)

    def build_rnn_cell(self, batch_size):

        rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnn_cell, self.attention_mechanism, attention_layer_size=self.dec_units)
        return rnn_cell

    def build_attention_mechanism(self, dec_units, memory, memory_sequence_length, attention_type='luong'):

        """from tensorflow documentation, you can use bahdanau in your project, luong works better in my case"""
        if attention_type == 'bahdanau':
            return tfa.seq2seq.BahdanauAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)
        else:
            return tfa.seq2seq.LuongAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)

    def build_initial_state(self, batch_size, encoder_state, dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_size, dtype=dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state

    def call(self, inputs, initial_state):
        x = self.embedding(inputs)
        outputs, _, _ = self.decoder(x, initial_state=initial_state, sequence_length=self.batch_size * [self.max_length_output-1])
        return outputs
