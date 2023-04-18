from pandas import DataFrame
from typing import Text, List, Optional, Union

from .multimedia_processing import MultimediaProcessing


class ComputerVision(MultimediaProcessing):
    def __init__(self, api_token):
        super().__init__(api_token)

    def image_classification(self, inputs: Union[Text, List], model: Optional[Text] = None) -> List:
        """
        Classify an image from a file path or an url.

        :param inputs: a string or a list of strings of the file paths or urls of the images to classify.
        :param model: the model to use for the image classification task. If not provided, the recommended model from Hugging Face will be used.
        :return: a list of dictionaries each containing the label and the confidence score for that label.
        """
        if type(inputs) == list:
            return self._query_in_list(inputs, model=model, task="image-classification")
        elif type(inputs) == str:
            return self._query(inputs, model=model, task="image-classification")

    def image_classification_in_df(self, df: DataFrame, column: Text, model: Optional[Text] = None) -> DataFrame:
        """
        Classify images from a dataframe.

        :param df: a pandas DataFrame containing the images to classify.
        :param column: the name of the column containing the file paths or urls of the images to classify.
        :param model: the model to use for the image classification task. If not provided, the recommended model from Hugging Face will be used.
        :return: a pandas DataFrame with the label for the images. Each label added will be the one with the highest confidence score for that particular image. The label will be added as a new column called 'predictions' to the original DataFrame.
        """
        predictions = self._query_in_df(df, column, model=model, task="image-classification")
        df["predictions"] = [prediction[0]['label'] for prediction in predictions]
        return df

    def object_detection(self, inputs: Union[Text, List], model: Optional[Text] = None) -> List:
        """
        Perform object detection on an image from a file path or an url.

        :param inputs: a string or a list of strings of the file paths or urls of the images to perform object detection on.
        :param model: the model to use for the object detection task. If not provided, the recommended model from Hugging Face will be used.
        :return: a list of dictionaries each containing the label, the confidence score for that label, and the bounding box coordinates.
        """
        if type(inputs) == list:
            return self._query(inputs, model=model, task="object-detection")
        elif type(inputs) == str:
            return self._query(inputs, model=model, task="object-detection")