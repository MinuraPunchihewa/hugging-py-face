import os
import unittest
import pandas as pd
from dotenv import load_dotenv
from pandas.testing import assert_frame_equal

from hugging_py_face.audio_processing import AudioProcessing

load_dotenv()


class TestAudioProcessingInDF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ap = AudioProcessing(os.environ.get("API_KEY"))

    def test_speech_recognition_in_df(self):
        paths = ["resources/amused.wav"]
        df = pd.DataFrame(paths, columns=['inputs'])

        assert_frame_equal(
            self.ap.speech_recognition_in_df(df, 'inputs'),
            pd.DataFrame(
                {
                    "inputs": ["resources/amused.wav"],
                    "predictions": ["I AM PLAYING A SINGLE HAND IN IT LOOKS LIKE A LOSING GAME"],
                }
            ),
        )

    def test_audio_classification_in_df(self):
        paths = ["resources/amused.wav"]
        df = pd.DataFrame(paths, columns=['inputs'])

        assert_frame_equal(
            self.ap.audio_classification_in_df(df, 'inputs'),
            pd.DataFrame(
                {
                    "inputs": ["resources/amused.wav"],
                    "predictions": ["hap"],
                }
            ),
        )