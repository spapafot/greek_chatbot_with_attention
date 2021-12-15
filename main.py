import ingest_data
import preprocessing
import numpy as np
import model
import os

filepath = "data"
all_files = os.walk(filepath)
with open("saved_files/character_names.txt", "r") as f:
    character_names = f.readlines()

character_names = list(map(lambda x: x.strip('\n'), character_names))
text_list = ingest_data.structure_files(all_files, filepath)
clean_list = ingest_data.create_clean_list(text_list, character_names)
VOCAB_SIZE = ingest_data.get_vocab_size(clean_list)
df = ingest_data.create_dataframe(clean_list)
df = df.dropna()

MAX_LEN = 25
EMBEDDING_DIM = 300

X = preprocessing.clean_text(df['input'].to_list())
y = preprocessing.clean_text(df['output'].to_list())
y_tagged = preprocessing.tag_start_end_sentences(y)

word2idx, idx2word, vocab = preprocessing.vocab_creator((X+y_tagged), VOCAB_SIZE)
encoder_sequences, decoder_sequences = preprocessing.text2seq(encoder_text=X, decoder_text=y_tagged, vocab_size=VOCAB_SIZE)
encoder_input_data, decoder_input_data = preprocessing.padding(encoder_sequences, decoder_sequences, MAX_LEN)
embedding_matrix = preprocessing.create_embedding_matrix(vocab, VOCAB_SIZE, EMBEDDING_DIM)
decoder_output_data = preprocessing.create_decoder_output(decoder_input_data, len(encoder_sequences), MAX_LEN, VOCAB_SIZE)
decoder_output_data.astype(dtype="float32")

embedding_layer = model.create_embedding_layer(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN, embedding_matrix)
model = model.create_seq2seq_model(EMBEDDING_DIM, VOCAB_SIZE, MAX_LEN, embedding_layer)

if __name__ == '__main__':
    print("GO")


"""
numpy.core._exceptions.MemoryError: Unable to allocate 23.5 GiB for an array with shape (12092, 25, 20828) and data type float32
https://stackoverflow.com/questions/57507832/unable-to-allocate-array-with-shape-and-data-type
"""