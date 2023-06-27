import os
import unittest
from dotenv import load_dotenv

from hugging_py_face.audio_processing import AudioProcessing
from hugging_py_face.exceptions import HTTPServiceUnavailableException

load_dotenv()


class TestAudioProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ap = AudioProcessing(os.environ.get("API_KEY"))
        cls.inputs = os.path.join(os.path.dirname(__file__), '..', 'resources', 'amused.wav')

    def assertAlmostEqualList(self, expected, actual, places=7):
        self.assertEqual(len(expected), len(actual))
        for exp, act in zip(expected, actual):
            self.assertEqual(exp['label'], act['label']), "Label values are not equal."
            self.assertAlmostEqual(exp['score'], act['score'], places=places), "Score values are not approximately equal."

    def test_automatic_speech_recognition(self):
        try:
            self.assertEqual(
                self.ap.automatic_speech_recognition(self.inputs),
                {
                    'text': 'I AM PLAYING A SINGLE HAND IN IT LOOKS LIKE A LOSING GAME'
                },
            )
        except HTTPServiceUnavailableException:
            self.assertRaises(HTTPServiceUnavailableException, lambda: self.ap.automatic_speech_recognition(self.inputs))

    def test_audio_classification(self):
        try:
            expected_result = [
                {'label': 'hap', 'score': 0.996896505355835},
                {'label': 'sad', 'score': 0.002958094235509634},
                {'label': 'neu', 'score': 9.905487240757793e-05},
                {'label': 'ang', 'score': 4.624627763405442e-05}
            ]

            actual_result = self.ap.audio_classification(self.inputs)
            self.assertAlmostEqualList(expected_result, actual_result)
        except HTTPServiceUnavailableException:
            self.assertRaises(HTTPServiceUnavailableException, lambda: self.ap.audio_classification(self.inputs))