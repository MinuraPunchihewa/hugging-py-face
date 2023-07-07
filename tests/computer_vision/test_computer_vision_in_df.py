import os
import unittest
import pandas as pd
from dotenv import load_dotenv
from pandas.testing import assert_frame_equal

from hugging_py_face.computer_vision import ComputerVision
from hugging_py_face.exceptions import HTTPServiceUnavailableException

load_dotenv()


class TestComputerVisionInDF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = ComputerVision(os.environ.get("API_KEY"))
        cls.inputs = [os.path.join(os.path.dirname(__file__), '..', 'resources', 'dogs.jpeg')]

    def test_image_classification_in_df(self):
        df = pd.DataFrame(self.inputs, columns=['inputs'])

        try:
            assert_frame_equal(
                self.cp.image_classification_in_df(df, 'inputs'),
                pd.DataFrame(
                    {
                        "inputs": self.inputs,
                        "predictions": ["golden retriever"],
                    }
                ),
                check_exact=False,
            )
        except HTTPServiceUnavailableException:
            pass