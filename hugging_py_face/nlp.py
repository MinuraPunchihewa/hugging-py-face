import json
import requests
from .config_parser import ConfigParser


class NLP:
    def __int__(self, api_token):
        self.api_token = api_token

        config_parser = ConfigParser()
        self.config = config_parser.get_config_dict()

    def _query(self, inputs, parameters=None, options=None, model=None, task=None):
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

    def text_classification(self, inputs, options=None, model=None):
        return self._query(inputs, options=options, model=model, task='text-classification')