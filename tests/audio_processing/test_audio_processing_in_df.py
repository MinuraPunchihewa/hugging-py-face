import os
import unittest
import pandas as pd
from dotenv import load_dotenv
from pandas.testing import assert_frame_equal

from hugging_py_face.audio_processing import AudioProcessing
from hugging_py_face.exceptions import HTTPServiceUnavailableException

load_dotenv()


class TestAudioProcessingInDF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ap = AudioProcessing(os.environ.get("API_KEY"))
        cls.inputs = [os.path.join(os.path.dirname(__file__), '..', 'resources', 'amused.wav')]

    def test_speech_recognition_in_df(self):
        df = pd.DataFrame(self.inputs, columns=['inputs'])

        try:
            assert_frame_equal(
                self.ap.speech_recognition_in_df(df, 'inputs'),
                pd.DataFrame(
                    {
                        "inputs": self.inputs,
                        "predictions": ["I AM PLAYING A SINGLE HAND IN IT LOOKS LIKE A LOSING GAME"],
                    }
                ),
            )
        except HTTPServiceUnavailableException:
            self.assertRaises(HTTPServiceUnavailableException, lambda: self.ap.speech_recognition_in_df(df, 'inputs'))

    def test_audio_classification_in_df(self):
        df = pd.DataFrame(self.inputs, columns=['inputs'])

        try:
            assert_frame_equal(
                self.ap.audio_classification_in_df(df, 'inputs'),
                pd.DataFrame(
                    {
                        "inputs": self.inputs,
                        "predictions": ["hap"],
                    }
                ),
            )
        except HTTPServiceUnavailableException:
            self.assertRaises(HTTPServiceUnavailableException, lambda: self.ap.audio_classification_in_df(df, 'inputs'))