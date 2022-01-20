import unittest

from src.project import NerClassifier


class AnonymizerTestCase(unittest.TestCase):
    def test_obfuscator_first_letter(self):
        obfuscation_mode = {"PER": {"first_letter": True}}
        entities = ["Cristiano", "Ronaldo"]
        labels = ["B-PER", "I-PER"]
        for elem, label in zip(entities, labels):
            mod_ent = NerClassifier.obfuscate_entity(elem, label, **obfuscation_mode)
            self.assertEqual(mod_ent, elem[0] + ".")

    def test_obfuscator_first_letter_loc(self):
        obfuscation_mode = {"PER": {"first_letter": True}, "LOC": {"first_letter": True}}
        entities = ["Cristiano", "Ronaldo", "Madrid"]
        labels = ["B-PER", "I-PER", "I-LOC"]
        for elem, label in zip(entities, labels):
            mod_ent = NerClassifier.obfuscate_entity(elem, label, **obfuscation_mode)
            self.assertEqual(mod_ent, elem[0] + ".")

    def test_obfuscator_replace(self):
        obfuscation_mode = {
            "PER": {"replace": "X"},
            "LOC": {"first_letter": True},
            "ORG": {"first_letter": True, "replace": "X"},
        }
        entities = ["Cristiano", "Ronaldo", "Madrid", "Football"]
        labels = ["B-PER", "I-PER", "B-LOC", "B-ORG"]
        mod_ent = NerClassifier.obfuscate_entity(entities[0], labels[0], **obfuscation_mode)
        self.assertEqual(mod_ent, "XXXXXXXXX")
        mod_ent = NerClassifier.obfuscate_entity(entities[1], labels[1], **obfuscation_mode)
        self.assertEqual(mod_ent, "XXXXXXX")
        mod_ent = NerClassifier.obfuscate_entity(entities[2], labels[2], **obfuscation_mode)
        self.assertEqual(mod_ent, "M.")
        mod_ent = NerClassifier.obfuscate_entity(entities[3], labels[3], **obfuscation_mode)
        self.assertEqual(mod_ent, "X.")


if __name__ == "__main__":
    unittest.main()
