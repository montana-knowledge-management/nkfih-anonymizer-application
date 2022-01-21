import nltk
import numpy as np
from digital_twin_distiller import ml_project
from digital_twin_distiller.encapsulator import Encapsulator
from digital_twin_distiller.ml_project import AbstractTask, Classifier
from digital_twin_distiller.modelpaths import ModelDir
from importlib_resources import files
from keras.models import load_model

from src.ner import DataLoader

nltk.download("punkt")


class nkfihAnonymizer(ml_project.MachineLearningProject):
    def custom_input(self, input_dict: dict) -> None:
        self._input_data.append(input_dict)

    def run(self):
        ner_task = NerTask(self._input_data, cache=self.cached_subtasks)
        result = ner_task.execute()
        self._output_data = [result]

    def cache(self):
        self.cached_subtasks["NER"] = NerClassifier()
        self.cached_subtasks["NER"].load_model()


class NerTask(AbstractTask):
    def define_subtasks(self, cache=None):
        self.classifier = cache["NER"]

    def execute(self):
        returnlist = list()
        for item in self._input_data:
            returnlist.append(self.classifier.run(item))
        if len(returnlist) == 1:
            returnlist = returnlist[0]
        return returnlist


class NerClassifier(Classifier):
    def load_model(self):
        self.classifier = load_model(
            files("resources")
            / "weighted_ner_epoch_1_with_trained_pretrained_embedding_bio_cells_20_dim_100_all_data.h5"
        )
        self.data_loader = DataLoader(max_sequence_length=60, use_lemmas=False, use_bio=True)
        self.data_loader.load_idx2word(files("resources") / "idx2word_punct_all.zip")
        self.data_loader.load_vocabulary(files("resources") / "vocab_punct_all.zip")
        self.data_loader.load_unique_labels_dict()

    @staticmethod
    def harvest_tags(labels: list, sentence: list, conversion_dict: dict):
        """
        Collects named entities from the given sentence.
        :param labels: list of label numbers
        :param sentence: list of tokens
        :param conversion_dict: dict for converting numeric labels to string labels
        :return: list of tuples containing extracted entities e.g. [("Cristiano Ronaldo", PER)]
        """
        # list for results
        tags = []
        # list of tuple entities
        state = []
        for label, token in zip(labels[: len(sentence)], sentence):
            label = conversion_dict.get(label)
            if label.startswith("O"):
                if state:
                    # appending tuple e.g ("Cristiano Ronaldo", PER)
                    tags.append((" ".join([elem[1] for elem in state]), state[0][0]))
                    state = []
            elif label.startswith("I-"):
                tag = label.split("I-")[-1]
                try:
                    if tag == state[-1][0]:
                        state.append((tag, token))
                except IndexError:
                    pass
            elif label.startswith("B-"):
                if state:
                    tags.append((" ".join([elem[1] for elem in state]), state[0][0]))
                    state = []
                tag = label.split("B-")[-1]
                state.append((tag, token))
        if state:
            tags.append((" ".join([elem[1] for elem in state]), state[0][0]))
        return tags

    @staticmethod
    def convert_ents_to_dict(entity_list: list):
        """
        Converts list of entities to a dictionary sorted by label types.
        :param entity_list: list of entities e.g. [("Lionel Messi", "PER"), ("Barcelona","LOC")]
        :return: {"PER": ["Lionel Messi"], "LOC": ["Barcelona"]}
        """
        result_dict = {}
        for ent, label in entity_list:
            if not result_dict.get(label):
                result_dict[label] = [ent]
            else:
                result_dict[label].append(ent)
        return result_dict

    @staticmethod
    def obfuscate_entity(entity, label, **modes_dict):
        """
        Modifies a recognized and taggged entity based on the given obfuscation rules.
        :param entity: entity, string
        :param label: label of the given entity
        :param modes_dict: dict containing modes for the obfuscation. Keys:
                The dict must contain entity types as keys: e.g. "PER". Under these types the following settings are available:
                    first_letter: boolean, converting entity to its starting letter e.g. Cristiano Ronaldo-> C.R.
                    replace: str, character to be used to replace, default: X
                example for modes_dict= {"PER":{"first_letter":True}, "ORG":{"replace":"X."}}
        :return: modified entity
        """
        main_labels = ["PER", "MISC", "LOC", "ORG"]
        for main_label in main_labels:
            if modes_dict.get(main_label):
                if main_label in label:
                    mode = modes_dict.get(main_label)
                    if "replace" in mode.keys():
                        replace_string = mode.get("replace")
                        if not replace_string:
                            replace_string = "X"
                        entity = replace_string * len(entity)
                    if mode.get("first_letter"):
                        entity = entity[0] + "."
        return entity

    def run(self, input_data):
        mode_dict = {"PER": {"first_letter": True, "replace": "X"}}
        text = input_data.get("text")
        preprocessed_input = self.data_loader.preprocess_new_data(
            text
        )  # [".", ",", ";", "!", "?", ":", "(", ")", "–"])
        # predicting class probabilities
        labels = self.classifier.predict(preprocessed_input)
        # choosing the tag with the highest probability
        y_predicted = np.argmax(labels, axis=-1)
        conversion_dict = {value: key for key, value in self.data_loader.unique_labels_dict.items()}
        y_labels = []
        extracted_entities = []
        anonymized_sentences = []
        for lab, sentence in zip(y_predicted, self.data_loader.tokens):
            # mapping non "O" labels to string containing the tag
            mapped_labels = [
                f"<{tok}; {conversion_dict.get(elem)}>"
                if conversion_dict.get(elem) != "O" and conversion_dict.get(elem) != None
                else tok
                for elem, tok in zip(lab[: len(sentence)], sentence)
            ]
            y_labels.extend(mapped_labels)
            obfuscated_labels = [
                f"{self.obfuscate_entity(tok, conversion_dict.get(elem), **mode_dict)}"
                if conversion_dict.get(elem) != "O" and conversion_dict.get(elem) != None
                else tok
                for elem, tok in zip(lab[: len(sentence)], sentence)
            ]
            # Collecting entities
            xtracted_ents = self.harvest_tags(labels=lab, sentence=sentence, conversion_dict=conversion_dict)
            extracted_entities.extend(xtracted_ents)
            anonymized_sentences.extend(obfuscated_labels)
        input_data.update({"Labeled": " ".join(y_labels).replace(" .", ".").replace(" ,", ",")})
        input_data.update({"Entities": self.convert_ents_to_dict(extracted_entities)})
        input_data.update({"Anonymized": " ".join(anonymized_sentences).replace(" .", ".").replace(" ,", ",")})
        return input_data


if __name__ == "__main__":
    ModelDir.set_base(__file__)
    app = nkfihAnonymizer(app_name="Anonymizer based on Distiller")  # , no_cache=True)
    server = Encapsulator(app)
    server.build_docs()
    server.run()
