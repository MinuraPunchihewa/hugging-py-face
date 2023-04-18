from pandas import DataFrame
from typing import Text, List, Dict, Optional, Union

from .multimedia_processing import MultimediaProcessing


class AudioProcessing(MultimediaProcessing):
    def __init__(self, api_token):
        super().__init__(api_token)

    def speech_recognition(self, inputs: Union[Text, List], model: Optional[Text] = None) -> Union[Dict, List]:
        """
        Perform speech recognition on an audio file from a file path or an url.

        :param inputs: a string or a list of strings of the file paths or urls of the audio files to perform speech recognition on.
        :param model: the model to use for the speech recognition task. If not provided, the recommended model from Hugging Face will be used.
        :return: a dictionary or a list of dictionaries containing the text recognized from the audio file(s).
        """
        if type(inputs) == list:
            return self._query_in_list(inputs, model=model, task="speech-recognition")
        elif type(inputs) == str:
            return self._query(inputs, model=model, task="speech-recognition")

    def speech_recognition_in_df(self, df: DataFrame, column: Text, model: Optional[Text] = None) -> DataFrame:
        """
        Perform speech recognition on audio files from a DataFrame.

        :param df: a pandas DataFrame containing the audio files to perform speech recognition on.
        :param column: the name of the column containing the file paths or urls of the audio files to perform speech recognition on.
        :param model: the model to use for the speech recognition task. If not provided, the recommended model from Hugging Face will be used.
        :return: a pandas DataFrame with the text recognized from the audio files. The text will be added as a new column called 'predictions' to the original DataFrame.
        """
        predictions = self._query_in_df(df, column, model=model, task="speech-recognition")
        df["predictions"] = [prediction['text'] for prediction in predictions]
        return df

    def audio_classification(self, inputs: Text, model: Optional[Text] = None) -> List:
        """
        Classify an audio file from a file path or an url.

        :param inputs: a string or a list of strings of the file paths or urls of the audio files to classify.
        :param model: the model to use for the audio classification task. If not provided, the recommended model from Hugging Face will be used.
        :return: a list of dictionaries or a list of lists of dictionaries each containing the label and the confidence score for that label.
        """
        if type(inputs) == list:
            return self._query_in_list(inputs, model=model, task="audio-classification")
        elif type(inputs) == str:
            return self._query(inputs, model=model, task="audio-classification")

    def audio_classification_in_df(self, df: DataFrame, column: Text, model: Optional[Text] = None) -> DataFrame:
        """
        Classify audio files from a DataFrame.

        :param df: a pandas DataFrame containing the audio files to classify.
        :param column: the name of the column containing the file paths or urls of the audio files to classify.
        :param model: the model to use for the audio classification task. If not provided, the recommended model from Hugging Face will be used.
        :return: a pandas DataFrame with the label for the audio files. Each label added will be the one with the highest confidence score for that particular audio file. The label will be added as a new column called 'predictions' to the original DataFrame.
        """
        predictions = self._query_in_df(df, column, model=model, task="audio-classification")
        df["predictions"] = [prediction[0]['label'] for prediction in predictions]
        return df