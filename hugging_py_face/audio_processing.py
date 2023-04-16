from typing import Text, List, Dict, Optional
from .multimedia_processing import MultimediaProcessing


class AudioProcessing(MultimediaProcessing):
    def __init__(self, api_token):
        super().__init__(api_token)

    def speech_recognition(self, input: Text, model: Optional[Text] = None) -> Dict:
        """
        Perform speech recognition on an audio file from a file path or an url.

        :param input: the file path or url to the audio file to perform speech recognition on
        :param model: the model to use for the speech recognition task. If not provided, the recommended model from Hugging Face will be used.
        :return: the text transcription of the audio file
        """
        return self._query(input, model=model, task="speech-recognition")

    def audio_classification(self, input: Text, model: Optional[Text] = None) -> List:
        """
        Classify an audio file from a file path or an url.

        :param input: the file path or url to the audio file to classify
        :param model: the model to use for the audio classification task. If not provided, the recommended model from Hugging Face will be used.
        :return: a list containing the labels and the confidence score for each label
        """
        return self._query(input, model=model, task="audio-classification")