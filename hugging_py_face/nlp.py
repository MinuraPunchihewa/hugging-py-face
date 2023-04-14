import json
import requests
from typing import Text, List, Dict, Optional, Union
from .config_parser import ConfigParser


class NLP:
    def __init__(self, api_token):
        self.api_token = api_token

        config_parser = ConfigParser()
        self.config = config_parser.get_config_dict()

    def _query(self, inputs: Union[Text, List, Dict], parameters: Dict = None, options: Dict = None, model: Text = None, task: Text = None) -> Dict:
        api_url = f"{self.config['BASE_URL']}/{model if model is not None else self.config['TASK_MODEL_MAP'][task]}"

        headers = {
            "Authorization": f"Bearer {self.api_token}"
        }

        data = {
            "inputs": inputs
        }

        if parameters is not None:
            data['parameters'] = parameters

        if options is not None:
            data['options'] = options

        response = requests.request("POST", api_url, headers=headers, data=json.dumps(data))
        return json.loads(response.content.decode("utf-8"))

    def fill_mask(self, inputs: Union[Text, List], options: Dict = None, model: Text = None) -> Dict:
        """
        Fill in a masked portion(token) of a string or a list of strings.

        :param inputs: a string or list of strings to be filled. Each input must contain the [MASK] token.
        :param options: a dict of options. For more information, see the `detailed parameters for the fill mask task <https://huggingface.co/docs/api-inference/detailed_parameters#fill-mask-task>`_.
        :param model: the model to use for the fill mask task. If not provided, the recommended model from Hugging Face will be used.
        :return: a list of dicts or a list of lists (of dicts) containing the possible completions and their associated probabilities.
        """
        return self._query(inputs, options=options, model=model, task='fill-mask')

    def summarization(self, inputs: Union[Text, List], parameters: Optional[Dict] = None, options: Optional[Dict] = None, model: Optional[Text] = None) -> Dict:
        """
        Summarize a string or a list of strings.

        :param inputs: a string or list of strings to be summarized.
        :param parameters: a dict of parameters. For more information, see the `detailed parameters for the summarization task <https://huggingface.co/docs/api-inference/detailed_parameters#summarization-task>`_.
        :param options: a dict of options. For more information, see the `detailed parameters for the summarization task <https://huggingface.co/docs/api-inference/detailed_parameters#summarization-task>`_.
        :param model: the model to use for the summarization task. If not provided, the recommended model from Hugging Face will be used.
        :return: a dict or a list of dicts of the summarized string(s).
        """
        return self._query(inputs, parameters=parameters, options=options, model=model, task='summarization')

    def text_classification(self, inputs: Union[Text, List], options: Dict = None, model: Text = None) -> Dict:
        """
        Analyze the sentiment of a string or a list of strings.

        :param inputs: a string or list of strings to be analyzed.
        :param options: a dict of options. For more information, see the `detailed parameters for the summarization task <https://huggingface.co/docs/api-inference/detailed_parameters#text-classification-task>`_.
        :param model: the model to use for the text classification task. If not provided, the recommended model from Hugging Face will be used.
        :return: a dict or a list of dicts indicating the sentiment of the string(s).
        """
        return self._query(inputs, options=options, model=model, task='text-classification')