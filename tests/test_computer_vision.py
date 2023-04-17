import os
import unittest
from dotenv import load_dotenv

from hugging_py_face.computer_vision import ComputerVision

load_dotenv()


class TestComputerVision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = ComputerVision(os.environ.get("API_KEY"))

    def test_image_classification(self):
        input = "resources/dogs.jpeg"

        self.assertEqual(
            self.cp.image_classification(input),
            [
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
            ],
        )

    def test_object_detection(self):
        input = "resources/dogs.jpeg"

        self.assertEqual(
            self.cp.object_detection(input),
            [
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
            ],
        )