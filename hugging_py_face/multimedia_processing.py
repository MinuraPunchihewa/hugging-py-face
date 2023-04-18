import json
import requests
from typing import Text, Dict, List, Optional, Union
from .config_parser import ConfigParser


class MultimediaProcessing:
    def __init__(self, api_token):
        self.api_token = api_token

        config_parser = ConfigParser()
        self.config = config_parser.get_config_dict()

    def _query(self, input: Text, model: Optional[Text] = None, task: Optional[Text] = None) -> Union[Dict, List]:
        api_url = f"{self.config['BASE_URL']}/{model if model is not None else self.config['TASK_MODEL_MAP'][task]}"

        headers = {
            "Authorization": f"Bearer {self.api_token}"
        }

        if input.startswith("http"):
            response = requests.get(input)
            response.raise_for_status()

            data = response.content

        else:
            with open(input, "rb") as f:
                data = f.read()

        response = requests.request("POST", api_url, headers=headers, data=data)
        return json.loads(response.content.decode("utf-8"))

    def _query_in_list(self, inputs: List[Text], model: Optional[Text] = None, task: Optional[Text] = None) -> List[Union[Dict, List]]:
        return [self._query(input, model, task) for input in inputs]

    def _query_in_df(self, df, input_column: Text, model: Optional[Text] = None, task: Optional[Text] = None) -> List[Union[Dict, List]]:
        return self._query_in_list(df[input_column].tolist(), model, task)