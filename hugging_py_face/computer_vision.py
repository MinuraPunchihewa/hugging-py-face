from typing import Text, List, Optional, Union
from .multimedia_processing import MultimediaProcessing


class ComputerVision(MultimediaProcessing):
    def __init__(self, api_token):
        super().__init__(api_token)

    def image_classification(self, inputs: Union[Text, List], model: Optional[Text] = None) -> List:
        """
        Classify an image from a file path or an url.

        :param inputs: a string or a list of strings of the file paths or urls of the images to classify
        :param model: the model to use for the image classification task. If not provided, the recommended model from Hugging Face will be used.
        :return: a list of dictionaries each containing the label and the confidence score for that label
        """
        if type(input) == list:
            return self._query_in_list(inputs, model=model, task="image-classification")
        elif type(input) == str:
            return self._query(inputs, model=model, task="image-classification")

    def object_detection(self, inputs: Union[Text, List], model: Optional[Text] = None) -> List:
        """
        Perform object detection on an image from a file path or an url.

        :param inputs: a string or a list of strings of the file paths or urls of the images to perform object detection on
        :param model: the model to use for the object detection task. If not provided, the recommended model from Hugging Face will be used.
        :return: a list of dictionaries each containing the label, the confidence score for that label, and the bounding box coordinates
        """
        if type(input) == list:
            return self._query(inputs, model=model, task="object-detection")
        elif type(input) == str:
            return self._query(inputs, model=model, task="object-detection")