import tensorflow as tf
from tensorflow.keras import optimizers

optimizer = optimizers.Adam()


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
def train_step(inp, targ, enc_hidden, encoder, decoder, optimizer, batch_size):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_h, enc_c = encoder(inp, enc_hidden)

        dec_input = targ[:, :-1]  # Ignore <end> token
        real = targ[:, 1:]  # ignore <start> token

        # Set the AttentionMechanism object with encoder_outputs
        decoder.attention_mechanism.setup_memory(enc_output)

        # Create AttentionWrapperState as initial_state for decoder
        decoder_initial_state = decoder.build_initial_state(batch_size, [enc_h, enc_c], tf.float32)
        pred = decoder(dec_input, decoder_initial_state)
        logits = pred.rnn_output
        loss = loss_function(real, logits)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss


def train_model(epochs, encoder, train_dataset):
    for epoch in range(epochs):

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0
        # print(enc_hidden[0].shape, enc_hidden[1].shape)

        for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss



