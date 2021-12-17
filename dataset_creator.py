import tensorflow as tf
from sklearn.model_selection import train_test_split
import unicodedata
import string
import re
import io


class Dataset:
    def __init__(self, filepath="saved_files/formatted.txt"):
        self.file_path = filepath
        self.inp_lang_tokenizer = None
        self.targ_lang_tokenizer = None

    def unicode_to_ascii(self, text):
        return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

    ## Step 1 and Step 2
    def preprocess_sentence(self, text):
        text = self.unicode_to_ascii(text.lower().strip())
        text = re.sub(r"([?.!,¿])", r" \1 ", text)
        text = re.sub(r'[…]', " ", text)
        text = re.sub(r'<[^<]+?>', ' ', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'[" "]+', " ", text)
        text = text.strip()
        text = '<start> ' + text + ' <end>'
        return text

    def create_dataset(self):
        lines = io.open(self.file_path, encoding='utf8').read().strip().split('\n')
        word_pairs = [[self.preprocess_sentence(w) for w in l.split('\t')] for l in lines]

        return zip(*word_pairs)

    # Step 3 and Step 4
    def tokenize(self, lang):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>')
        lang_tokenizer.fit_on_texts(lang)
        tensor = lang_tokenizer.texts_to_sequences(lang)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

        return tensor, lang_tokenizer

    def load_dataset(self):
        # creating cleaned input, output pairs
        targ_lang, inp_lang = self.create_dataset()
        total_lines = len(targ_lang)

        input_tensor, inp_lang_tokenizer = self.tokenize(inp_lang)
        target_tensor, targ_lang_tokenizer = self.tokenize(targ_lang)

        return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer, total_lines

    def call(self, BUFFER_SIZE, BATCH_SIZE):
        input_tensor, target_tensor, self.inp_lang_tokenizer, self.targ_lang_tokenizer, total_lines = self.load_dataset()

        X_train, X_test, y_train, y_test = train_test_split(input_tensor, target_tensor, test_size=0.2)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

        return train_dataset, val_dataset, self.inp_lang_tokenizer, self.targ_lang_tokenizer, total_lines