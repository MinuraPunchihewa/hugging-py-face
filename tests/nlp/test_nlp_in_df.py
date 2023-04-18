import os
import unittest
import pandas as pd
from dotenv import load_dotenv
from pandas.testing import assert_frame_equal

from hugging_py_face.nlp import NLP

load_dotenv()


class TestNLPInDF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = NLP(os.environ.get("API_KEY"))

    def test_text_classification_in_df(self):
        texts = ["I like you. I love you", "I don't like you. I hate you"]
        df = pd.DataFrame(texts, columns=['texts'])

        assert_frame_equal(
            self.nlp.text_classification_in_df(df, 'texts'),
            pd.DataFrame(
                {
                    "texts": ["I like you. I love you", "I don't like you. I hate you"],
                    "predictions": ["POSITIVE", "NEGATIVE"],
                }
            ),
        )