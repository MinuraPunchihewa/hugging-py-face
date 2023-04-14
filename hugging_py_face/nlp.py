import json
import requests
from typing import Text, List, Dict, Union
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
        return self._query(inputs, options=options, model=model, task='fill-mask')

    def summarization(self, inputs: Union[Text, List], parameters: Dict = None, options: Dict = None, model: Text = None) -> Dict:
        return self._query(inputs, parameters=parameters, options=options, model=model, task='summarization')

    def text_classification(self, inputs: Union[Text, List], options: Dict = None, model: Text = None) -> Dict:
        return self._query(inputs, options=options, model=model, task='text-classification')