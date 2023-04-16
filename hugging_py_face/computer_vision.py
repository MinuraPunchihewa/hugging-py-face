import json
import requests
from typing import Text, List, Dict, Optional, Union
from .multimedia_processing import MultimediaProcessing


class ComputerVision(MultimediaProcessing):
    def __init__(self, api_token):
        super().__init__(api_token)

    def image_classification(self, input: Text, model: Optional[Text] = None) -> List:
        """
        :param input: the file path or url to the image to classify
        :param model: the model to use for the image classification task. If not provided, the recommended model from Hugging Face will be used.
        :return: a list of dictionaries each containing the label and the confidence score for that label
        """
        return self._query(input, model=model, task="image-classification")

    def object_detection(self, input: Text, model: Optional[Text] = None) -> List:
        """
        :param input: the file path or url to the image to perform object detection on
        :param model: the model to use for the object detection task. If not provided, the recommended model from Hugging Face will be used.
        :return: a list of dictionaries each containing the label, the confidence score for that label, and the bounding box coordinates
        """
        return self._query(input, model=model, task="object-detection")