import os
import unittest
from dotenv import load_dotenv

from hugging_py_face.audio_processing import AudioProcessing

load_dotenv()


class TestAudioProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ap = AudioProcessing(os.environ.get("API_KEY"))

    def test_speech_recognition(self):
        input = "resources/amused.wav"

        self.assertEqual(
            self.ap.speech_recognition(input),
            {
                'text': 'I AM PLAYING A SINGLE HAND IN IT LOOKS LIKE A LOSING GAME'
            },
        )

    def test_audio_classification(self):
        input = "resources/amused.wav"

        self.assertEqual(
            self.ap.audio_classification(input),
            [
                {
                    'score': 0.996896505355835,
                    'label': 'hap'
                },
                {
                    'score': 0.0029580998234450817,
                    'label': 'sad'
                },
                {
                    'score': 9.905469050863758e-05,
                    'label': 'neu'
                },
                {
                    'score': 4.624614666681737e-05,
                    'label': 'ang'
                }
            ],
        )