from att_model import Encoder, Decoder
from dataset_creator import Dataset
from utils import create_embedding_matrix
from tensorflow.keras import optimizers
import tensorflow as tf
import tensorflow_addons as tfa
import os

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# with open("saved_files/character_names.txt", "r") as f:
#     character_names = f.readlines()


# Play with these, did not see satisfying results before the 40 epoch mark though
# embedding_dim is 300 because i am using Spacy's pretrained vectors

BUFFER_SIZE = 32000
BATCH_SIZE = 64
embedding_dim = 300
units = 768
EPOCHS = 50


def loss_function(real, pred):
    # real shape = (BATCH_SIZE, max_length_output)
    # prediction shape = (BATCH_SIZE, max_length_output, tar_vocab_size )

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


# Recommend beam, got decent results

def beam_evaluate_sentence(sentence, beam_width=10):
    sentence = dataset.preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_input,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)
    inference_batch_size = inputs.shape[0]
    result = ''

    enc_start_state = [tf.zeros((inference_batch_size, units)), tf.zeros((inference_batch_size, units))]
    enc_out, enc_h, enc_c = encoder_model(inputs, enc_start_state)

    dec_h = enc_h
    dec_c = enc_c

    start_tokens = tf.fill([inference_batch_size], targ_lang.word_index['<start>'])
    end_token = targ_lang.word_index['<end>']

    # From official documentation
    # NOTE If you are using the BeamSearchDecoder with a cell wrapped in AttentionWrapper, then you must ensure that:
    # The encoder output has been tiled to beam_width via tfa.seq2seq.tile_batch (NOT tf.tile).
    # The batch_size argument passed to the get_initial_state method of this wrapper is equal to true_batch_size * beam_width.
    # The initial state created with get_initial_state above contains a cell_state value containing properly tiled final state from the encoder.

    enc_out = tfa.seq2seq.tile_batch(enc_out, multiplier=beam_width)
    decoder_model.attention_mechanism.setup_memory(enc_out)
    print("beam_with * [batch_size, max_length_input, rnn_units] :  3 * [1, 16, 1024]] :", enc_out.shape)

    # set decoder_inital_state which is an AttentionWrapperState considering beam_width
    hidden_state = tfa.seq2seq.tile_batch([enc_h, enc_c], multiplier=beam_width)
    decoder_initial_state = decoder_model.rnn_cell.get_initial_state(batch_size=beam_width * inference_batch_size,
                                                                     dtype=tf.float32)
    decoder_initial_state = decoder_initial_state.clone(cell_state=hidden_state)

    # Instantiate BeamSearchDecoder
    decoder_instance = tfa.seq2seq.BeamSearchDecoder(decoder_model.rnn_cell, beam_width=beam_width,
                                                     output_layer=decoder_model.fc)
    decoder_embedding_matrix = decoder_model.embedding.variables[0]

    # The BeamSearchDecoder object's call() function takes care of everything.
    outputs, final_state, sequence_lengths = decoder_instance(decoder_embedding_matrix, start_tokens=start_tokens,
                                                              end_token=end_token, initial_state=decoder_initial_state)
    # outputs is tfa.seq2seq.FinalBeamSearchDecoderOutput object.
    # The final beam predictions are stored in outputs.predicted_id
    # outputs.beam_search_decoder_output is a tfa.seq2seq.BeamSearchDecoderOutput object which keep tracks of beam_scores and parent_ids while performing a beam decoding step
    # final_state = tfa.seq2seq.BeamSearchDecoderState object.
    # Sequence Length = [inference_batch_size, beam_width] details the maximum length of the beams that are generated

    # outputs.predicted_id.shape = (inference_batch_size, time_step_outputs, beam_width)
    # outputs.beam_search_decoder_output.scores.shape = (inference_batch_size, time_step_outputs, beam_width)
    # Convert the shape of outputs and beam_scores to (inference_batch_size, beam_width, time_step_outputs)
    final_outputs = tf.transpose(outputs.predicted_ids, perm=(0, 2, 1))
    beam_scores = tf.transpose(outputs.beam_search_decoder_output.scores, perm=(0, 2, 1))

    return final_outputs.numpy(), beam_scores.numpy()


def beam_translate(sentence):
    result, beam_scores = beam_evaluate_sentence(sentence)
    print(result.shape, beam_scores.shape)
    for beam, score in zip(result, beam_scores):
        print(beam.shape, score.shape)
        output = targ_lang.sequences_to_texts(beam)
        output = [a[:a.index('<end>')] for a in output]
        beam_score = [a.sum() for a in score]
        print('Input: %s' % (sentence))
        for i in range(len(output)):
            print('{} Predicted translation: {}  {}'.format(i + 1, output[i], beam_score[i]))


# Did not

def evaluate_sentence(sentence):
    sentence = dataset.preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_input,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)
    inference_batch_size = inputs.shape[0]
    result = ''

    enc_start_state = [tf.zeros((inference_batch_size, units)), tf.zeros((inference_batch_size, units))]
    enc_out, enc_h, enc_c = encoder_model(inputs, enc_start_state)

    dec_h = enc_h
    dec_c = enc_c

    start_tokens = tf.fill([inference_batch_size], targ_lang.word_index['<start>'])
    end_token = targ_lang.word_index['<end>']

    greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

    # Instantiate BasicDecoder object
    decoder_instance = tfa.seq2seq.BasicDecoder(cell=decoder_model.rnn_cell, sampler=greedy_sampler,
                                                output_layer=decoder_model.fc)
    # Setup Memory in decoder stack
    decoder_model.attention_mechanism.setup_memory(enc_out)

    # set decoder_initial_state
    decoder_initial_state = decoder_model.build_initial_state(inference_batch_size, [enc_h, enc_c], tf.float32)

    ### Since the BasicDecoder wraps around Decoder's rnn cell only, you have to ensure that the inputs to BasicDecoder
    ### decoding step is output of embedding layer. tfa.seq2seq.GreedyEmbeddingSampler() takes care of this.
    ### You only need to get the weights of embedding layer, which can be done by decoder.embedding.variables[0] and pass this callabble to BasicDecoder's call() function

    decoder_embedding_matrix = decoder_model.embedding.variables[0]

    outputs, _, _ = decoder_instance(decoder_embedding_matrix, start_tokens=start_tokens, end_token=end_token,
                                     initial_state=decoder_initial_state)
    return outputs.sample_id.numpy()


def translate(sentence):
    result = evaluate_sentence(sentence)
    print(result)
    result = targ_lang.sequences_to_texts(result)
    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))



if __name__ == '__main__':

    dataset = Dataset()
    targ_lang, inp_lang = dataset.create_dataset()
    train_dataset, val_dataset, tokenizer, total_lines = dataset.call(BUFFER_SIZE, BATCH_SIZE)

    vocab_size = len(tokenizer.word_index) + 1

    steps_per_epoch = total_lines // BATCH_SIZE
    input_batch, target_batch = next(iter(train_dataset))
    max_length_input = input_batch.shape[1]
    max_length_output = target_batch.shape[1]

    embedding_matrix = create_embedding_matrix(tokenizer.word_index, vocab_size, embedding_dim)

    encoder_model = Encoder(vocab_size=vocab_size,
                            embedding_dim=embedding_dim,
                            enc_units=units,
                            batch_size=BATCH_SIZE,
                            embedding_matrix=embedding_matrix,
                            max_length_input=max_length_input)

    decoder_model = Decoder(vocab_size=vocab_size,
                            embedding_dim=embedding_dim,
                            dec_units=units,
                            batch_size=BATCH_SIZE,
                            max_length_input=max_length_input,
                            max_length_output=max_length_output,
                            embedding_matrix=embedding_matrix)

    optimizer = optimizers.Adam(learning_rate=0.0002)

    checkpoint_dir = '/content/checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder_model, decoder=decoder_model)

    for epoch in range(EPOCHS):

        enc_hidden = encoder_model.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch)):

            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss
            if batch % 10 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))


    """ Play around with beam_translate and translate() """
    while True:
        usr_inp = input()
        if usr_inp == "q":
            break
        else:
            beam_translate(usr_inp)
            translate(usr_inp)