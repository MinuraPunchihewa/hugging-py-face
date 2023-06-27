import os
import unittest
from dotenv import load_dotenv

from hugging_py_face.computer_vision import ComputerVision
from hugging_py_face.exceptions import HTTPServiceUnavailableException

load_dotenv()


class TestComputerVision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = ComputerVision(os.environ.get("API_KEY"))
        cls.inputs = os.path.join(os.path.dirname(__file__), '..', 'resources', 'dogs.jpeg')

    def assertAlmostEqualList(self, expected, actual, places=7):
        self.assertEqual(len(expected), len(actual))
        for exp, act in zip(expected, actual):
            self.assertEqual(exp['label'], act['label']), "Label values are not equal."
            self.assertAlmostEqual(exp['score'], act['score'], places=places), "Score values are not approximately equal."

            if 'box' in exp and 'box' in act:
                assert exp['box'] == act['box'], "Box values are not equal."
            elif 'box' in exp:
                raise AssertionError("Box field is missing in actual result.")
            elif 'box' in act:
                raise AssertionError("Box field is missing in expected result.")

    def test_image_classification(self):
        try:
            expected_result = [
                {
                    'score': 0.9061778783798218,
                    'label': 'golden retriever'
                },
                {
                    'score': 0.06364733725786209,
                    'label': 'Labrador retriever'
                },
                {
                    'score': 0.005189706105738878,
                    'label': 'Sussex spaniel'
                },
                {
                    'score': 0.0026904833503067493,
                    'label': 'clumber, clumber spaniel'
                },
                {
                    'score': 0.0026738110464066267,
                    'label': 'cocker spaniel, English cocker spaniel, cocker'
                }
            ]
            actual_result = self.cp.image_classification(self.inputs)
            self.assertAlmostEqualList(expected_result, actual_result, places=6)
        except HTTPServiceUnavailableException:
            self.assertRaises(HTTPServiceUnavailableException, lambda: self.cp.image_classification(self.inputs))

    def test_object_detection(self):
        try:
            expected_result = [
                {
                    'score': 0.9990463852882385,
                    'label': 'dog',
                    'box': {'xmin': 1329, 'ymin': 961, 'xmax': 2668, 'ymax': 3149}
                },
                {
                    'score': 0.9985553622245789,
                    'label': 'dog',
                    'box': {'xmin': 2598, 'ymin': 827, 'xmax': 3902, 'ymax': 3190}
                }
            ]
            actual_result = self.cp.object_detection(self.inputs)

            self.assertAlmostEqualList(expected_result, actual_result, places=6)
        except HTTPServiceUnavailableException:
            self.assertRaises(HTTPServiceUnavailableException, lambda: self.cp.object_detection(self.inputs))
