import os
import unittest
import pandas as pd
from dotenv import load_dotenv
from pandas.testing import assert_frame_equal

from hugging_py_face.computer_vision import ComputerVision

load_dotenv()


class TestComputerVisionInDF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = ComputerVision(os.environ.get("API_KEY"))

    def test_image_classification_in_df(self):
        paths = ["resources/dogs.jpeg"]
        df = pd.DataFrame(paths, columns=['inputs'])

        assert_frame_equal(
            self.cp.image_classification_in_df(df, 'inputs'),
            pd.DataFrame(
                {
                    "inputs": ["resources/dogs.jpeg"],
                    "predictions": ["golden retriever"],
                }
            ),
        )