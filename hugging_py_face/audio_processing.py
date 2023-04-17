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
        if type(input) == list:
            return self._query_in_list(inputs, model=model, task="speech-recognition")
        elif type(input) == str:
            return self._query(input, model=model, task="speech-recognition")

    def audio_classification(self, inputs: Text, model: Optional[Text] = None) -> List:
        """
        Classify an audio file from a file path or an url.

        :param inputs: a string or a list of strings of the file paths or urls of the audio files to classify.
        :param model: the model to use for the audio classification task. If not provided, the recommended model from Hugging Face will be used.
        :return: a list of dictionaries or a list of lists of dictionaries each containing the label and the confidence score for that label.
        """
        if type(input) == list:
            return self._query_in_list(inputs, model=model, task="audio-classification")
        elif type(input) == str:
            return self._query(input, model=model, task="audio-classification")