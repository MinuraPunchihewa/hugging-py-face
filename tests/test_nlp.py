import os
import unittest
from dotenv import load_dotenv

from hugging_py_face.nlp import NLP

load_dotenv()


class TestNLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = NLP(os.environ.get("API_KEY"))

    def test_text_classification(self):
        text = "I like you. I love you"

        self.assertEqual(
            self.nlp.text_classification(text),
            [
                [
                    {"label": "POSITIVE", "score": 0.9998738765716553},
                    {"label": "NEGATIVE", "score": 0.00012611244164872915},
                ]
            ],
        )
