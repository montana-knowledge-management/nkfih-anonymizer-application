import csv
from collections import Counter

import fasttext
import numpy as np
from digital_twin_distiller.text_readers import JsonReader
from digital_twin_distiller.text_writers import JsonWriter
from hungarian_stemmer.hungarian_stemmer import HungarianStemmer
from importlib_resources import files
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize, word_tokenize
from tensorflow.keras.utils import to_categorical


class DataLoader:
    tokens = []
    labels = []
    lemmas = []
    unique_labels = []

    def __init__(self, max_sequence_length=200, use_lemmas=False, use_bio=False):
        self.padding_settings = {"maxlen": max_sequence_length, "padding": "post", "truncating": "post", "value": 0}
        self.use_lemmas = use_lemmas
        self.use_bio = use_bio
        if use_lemmas:
            self.load_hunspell()

    def load_training_file(self, number, filter_pos: list = None):
        """
        Loads a specific tsv file for training.
        :param number: number of the training tsv fileto be usedduring loading
        :param filter_pos: POS tags to be filtered from training data in list format.
        :return:
        """
        print(f"Loading huwiki.{number}")
        sentence = []
        label = []
        lemma = []
        unique_labels = set()
        tsv_file = open(str(files("resources") / f"huwiki.{number}.tsv"))
        read_tsv = csv.reader(tsv_file, delimiter="\t", quoting=csv.QUOTE_NONE)
        counter = Counter()
        if not filter_pos:
            filter_pos = {}
        else:
            filter_pos = set(filter_pos)

        for row in read_tsv:
            if not row:
                self.tokens.append(sentence)
                self.lemmas.append(lemma)
                self.labels.append(label)
                unique_labels = unique_labels.union(set(label))
                counter.update(label)
                sentence = []
                label = []
                lemma = []
            else:
                pos_tag = row[3]
                # filtering POS tag
                if not {pos_tag}.intersection(filter_pos):
                    if not self.use_lemmas:
                        sentence.append(row[0])
                    else:
                        lemma.append(row[-2])
                    tag = row[-1]
                    # if BIO tagging is not important keeping only the tag e.g. B-PER -> PER
                    if not self.use_bio:
                        if "-" in tag:
                            tag = tag.split("-")[-1]
                    label.append(tag)
        print(unique_labels)
        print(f"Number of labels: {counter.most_common(10)}")
        self.unique_labels = unique_labels

    def load_unique_labels_dict(self):
        """
        Loads unique labels dict.
        :return:
        """
        if self.use_bio:
            self.unique_labels_dict = JsonReader().read(files("resources") / "unique_labels_dict_bio.json")
        else:
            self.unique_labels_dict = JsonReader().read(files("resources") / "unique_labels_dict.json")

    def load_all_training_data(self, filter_pos: list = None):
        """
        Loads all training file.
        :param filter_pos: POS tags that are filtered during loading
        :return: None
        """
        for idx in range(1, 5):
            self.load_training_file(number=idx, filter_pos=filter_pos)

    def convert_unique_labels_to_dict(self):
        self.unique_labels_dict = {}
        for idx, elem in enumerate(self.unique_labels):
            self.unique_labels_dict[elem] = idx + 1

    def convert_labels_to_ids(self):
        """
        Converts text labels to ids in the dictionary stored by unique_labels_dict.
        :return:
        """
        converted_labels = []
        for doc in self.labels:
            converted_doc_label = []
            for label in doc:
                converted_doc_label.append(self.unique_labels_dict.get(label))
            converted_labels.append(converted_doc_label)
        return converted_labels

    def convert_labels_to_categorical(self, converted_labels):
        """
        Converts numeric labels into one-hot encoded vectors.
        :param converted_labels: labels in number format, not text
        :return:
        """
        self.labels = to_categorical(converted_labels)
        return self.labels

    def build_vocab(self, tokenized_texts=None):
        """
        Builds a vocabulary from the tokenized input data.
        :param tokenized_texts: list of list of strings
        :return:
        """
        if not tokenized_texts:
            if self.use_lemmas:
                tokenized_texts = self.lemmas
            else:
                tokenized_texts = self.tokens
        all_tokens = []
        vocabulary = {}
        index_to_word = {}
        for document in tokenized_texts:
            all_tokens.extend(document)
        all_tokens = list(set(all_tokens))
        for idx, token in enumerate(all_tokens):
            # ensuring 0 index for padding and 1 index for [UNK] token
            vocabulary[token] = idx + 2
            index_to_word[idx + 2] = token
        self.vocabulary = vocabulary
        self.index_to_word = index_to_word
        return vocabulary, index_to_word

    def get_training_data(self):
        """
        Returns training data.
        :param lemmatized: whether to return lemmatized data or original.
        :return: list of list of tokens and list of list of labels
        """
        if self.use_lemmas:
            return self.lemmas, self.labels
        else:
            return self.tokens, self.labels

    def save_vocabulary(self, path_to_save):
        """
        Save vocabulary (e. g. {"egy":1, "kis":2, "malac": 3}) from the input text to json.
        :param path_to_save:
        :return:
        """
        JsonWriter().write(self.vocabulary, path_to_save)

    def save_idx2word(self, path_to_save):
        """
        Save index to word dictionary (e. g. {1: 'egy', 2: 'kis', 3: 'malac'}) to json.
        :param path_to_save:
        :return:
        """
        JsonWriter().write(self.index_to_word, path_to_save)

    def load_vocabulary(self, path_to_load_from):
        """
        Loads pre-build vocabulary from json file.
        :param path_to_load_from:
        :return:
        """
        self.vocabulary = JsonReader().read(path_to_load_from)

    def load_idx2word(self, path_to_load_from):
        """
        Loads pre-built index-to-word dictionary from json.
        :param path_to_load_from:
        :return:
        """
        # all index keys are strings now
        loaded_json = JsonReader().read(path_to_load_from)
        # converting strings to integers
        self.index_to_word = {int(key): value for key, value in loaded_json.items()}

    def load_fasttext_model(self, fasttext_model_path):
        """
        Loads models from https://fasttext.cc/docs/en/crawl-vectors.html. Only bin format is supported!
        :param fasttext_model_path: path for pretrained model in bin format is supported.
        :return:
        """
        self.fasttext_model = fasttext.load_model(fasttext_model_path)
        return self.fasttext_model

    def convert_documents_to_ids(self):
        """
        Converts tokens in the documents into ids based on the pre-built vocabulary.
        :return:
        """
        converted_documents = []
        if self.use_lemmas:
            docs = self.lemmas
        else:
            docs = self.tokens
        for document in docs:
            converted_document = []
            for token in document:
                index = self.vocabulary.get(token)
                if not index:
                    # Out Of Vocabulary words have index 1
                    index = 1
                converted_document.append(index)
            converted_documents.append(converted_document)
        return converted_documents

    def pad_to_length(self, documents):
        """
        Performs padding and truncation on documents.
        :param sequence:
        :param kwargs:
            sequences: List of sequences (each sequence is a list of integers).
            maxlen: Optional Int, maximum length of all sequences. If not provided, sequences will be padded to the length
            of the longest individual sequence.
            dtype: (Optional, defaults to int32). Type of the output sequences. To pad sequences with variable length
            strings, you can use object.
            padding: String, 'pre' or 'post' (optional, defaults to 'pre'): pad either before or after each sequence.
            truncating: String, 'pre' or 'post' (optional, defaults to 'pre'): remove values from sequences larger than
            maxlen, either at the beginning or at the end of the sequences.
            value: Float or String, padding value. (Optional, defaults to 0.)

        :return:
        """
        padded_documents = np.array(pad_sequences(documents, **self.padding_settings))
        return padded_documents

    def load_hunspell(self):
        hunstem = HungarianStemmer()
        self.hunspell_obj = hunstem

    def _tokenize(self, new_data: str, filter_tokens: list = None):
        if filter_tokens is None:
            filter_tokens = []
        tokens = [word_tokenize(sentence) for sentence in sent_tokenize(new_data)]
        if self.use_lemmas:
            preprocessed_data = []
            for sentence in tokens:
                stemmed = [
                    self.hunspell_obj.stem(token)[0]
                    for token in sentence
                    if token not in filter_tokens and self.hunspell_obj.stem(token)
                ]
                preprocessed_data.append(stemmed)
        else:
            preprocessed_data = []
            for sentence in tokens:
                stemmed = [token for token in sentence if token not in filter_tokens]
                preprocessed_data.append(stemmed)
        if self.use_lemmas:
            self.lemmas = preprocessed_data
        else:
            self.tokens = preprocessed_data

    def preprocess_new_data(self, new_data: str, filter_tokens: list = None):
        """
        Preprocesses unseen data.
        :param new_data: Document to be examined as a string.
        :param filter_tokens: list of tokens to be filt
        :return: preprocessed data list of tokens
        """
        self._tokenize(new_data=new_data, filter_tokens=filter_tokens)
        converted_docs = self.convert_documents_to_ids()
        converted_docs = self.pad_to_length(converted_docs)
        return converted_docs

    def create_pretrained_embedding_matrix(self, use_all_pretrained=False):
        """
        Creates embedding matrix based on the training data vocabulary and the whole vocabulary of the pre-trained
        fasttext model.
        :return: embedding model
        """
        if use_all_pretrained:
            # words not present in fasttext model
            additional_fasttext_vocab = set(self.fasttext_model.get_words()).difference(set(self.vocabulary.keys()))
        original_vocab_size = len(self.index_to_word)
        if use_all_pretrained:
            # size of vocabulary is the size of the fasttext model plus the extra words in the training data
            vocab_size = original_vocab_size + len(additional_fasttext_vocab)
        else:
            vocab_size = original_vocab_size
        vector_dim = self.fasttext_model.get_dimension()
        # size of the index_to_word dictionary is incremented by the size of the vocab by 2 to handle [PAD] and [UNK] tokens
        # with index 0 and 1
        embedding_matrix = np.zeros((vocab_size + 2, vector_dim))
        # adding
        for i in range(vocab_size - 2):
            embedding_vector = self.fasttext_model.get_word_vector(self.index_to_word[i + 2])
            if embedding_vector is not None:
                embedding_matrix[i + 2] = embedding_vector
        if use_all_pretrained:
            for idx, word in enumerate(additional_fasttext_vocab):
                embedding_vector = self.fasttext_model.get_word_vector(word)
                if embedding_vector is not None:
                    embedding_matrix[idx + vocab_size + 2] = embedding_vector

        return embedding_matrix


if __name__ == "__main__":
    vocab_path = str(files("resources") / "vocab_punct_all.json")
    idx2word_path = str(files("resources") / "idx2word_punct_all.json")
    loader = DataLoader(use_lemmas=False, max_sequence_length=5, use_bio=True)
    # loader.load_training_file(4)#, filter_pos=["PUNCT"])
    loader.load_all_training_data()
    loader.load_unique_labels_dict()
    loader.build_vocab()
    loader.save_vocabulary(vocab_path)
    loader.save_idx2word(idx2word_path)
