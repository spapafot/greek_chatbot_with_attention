import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from att_model import Encoder, Decoder
from dataset_creator import Dataset
from utils import create_embedding_matrix
from tensorflow.keras import optimizers
import tensorflow as tf


with open("saved_files/character_names.txt", "r") as f:
    character_names = f.readlines()

BUFFER_SIZE = 32000
BATCH_SIZE = 64
num_examples = 1000
max_len = 25
embedding_dim = 300
units = 1024

dataset = Dataset()

train_dataset, val_dataset, inp_lang, targ_lang = dataset.call(num_examples, BUFFER_SIZE, BATCH_SIZE)

vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1
steps_per_epoch = num_examples // BATCH_SIZE
example_input_batch, example_target_batch = next(iter(train_dataset))
max_length_input = example_input_batch.shape[1]
max_length_output = example_target_batch.shape[1]


embedding_matrix = create_embedding_matrix(inp_lang.word_index, vocab_inp_size, embedding_dim)

encoder_model = Encoder(vocab_size=vocab_inp_size,
                        embedding_dim=embedding_dim,
                        enc_units=units,
                        batch_size=BATCH_SIZE,
                        embedding_matrix=embedding_matrix,
                        max_length_input=max_length_input)

decoder_model = Decoder(vocab_size=vocab_tar_size,
                        embedding_dim=embedding_dim,
                        dec_units=units,
                        batch_size=BATCH_SIZE,
                        max_length_input=max_length_input,
                        max_length_output=max_length_output)

optimizer = optimizers.Adam(learning_rate=0.0002)


def loss_function(real, pred):
    # real shape = (BATCH_SIZE, max_length_output)
    # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = cross_entropy(y_true=real, y_pred=pred)
    mask = tf.logical_not(tf.math.equal(real, 0))  # output 0 for y=0 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = mask * loss
    loss = tf.reduce_mean(loss)
    return loss


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_h, enc_c = encoder_model(inp, enc_hidden)

        dec_input = targ[:, :-1]  # Ignore <end> token
        real = targ[:, 1:]  # ignore <start> token

        # Set the AttentionMechanism object with encoder_outputs
        decoder_model.attention_mechanism.setup_memory(enc_output)

        # Create AttentionWrapperState as initial_state for decoder
        decoder_initial_state = decoder_model.build_initial_state(BATCH_SIZE, [enc_h, enc_c], tf.float32)
        pred = decoder_model(dec_input, decoder_initial_state)
        logits = pred.rnn_output
        loss = loss_function(real, logits)

    variables = encoder_model.trainable_variables + decoder_model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss


EPOCHS = 10

for epoch in range(EPOCHS):

    enc_hidden = encoder_model.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch)):

        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss
        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))

if __name__ == '__main__':
    print("GO")
