import json
import requests
from typing import Text, List, Dict, Optional, Union
from .multimedia_processing import MultimediaProcessing


class ComputerVision(MultimediaProcessing):
    def __init__(self, api_token):
        super().__init__(api_token)

    def image_classification(self, input: Text, model: Optional[Text] = None) -> List:
        """
        :param input:
        :param model:
        :return:
        """
        return self._query(input, model=model, task="image-classification")

    def object_detection(self, input: Text, model: Optional[Text] = None) -> List:
        """
        :param input:
        :param model:
        :return:
        """
        return self._query(input, model=model, task="object-detection")