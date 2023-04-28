import json
import time
import logging
import requests
import logging.config
from typing import Text, Dict, List, Optional, Union

from .config_parser import ConfigParser
from .exceptions import HTTPServiceUnavailableException

logging_config_parser = ConfigParser('config/logging.yaml')
logging.config.dictConfig(logging_config_parser.get_config_dict())
logger = logging.getLogger()


class MultimediaProcessing:
    def __init__(self, api_token):
        self.api_token = api_token

        config_parser = ConfigParser()
        self.config = config_parser.get_config_dict()

        self.logger = logger

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

        retries = 0

        while retries < self.config['MAX_RETRIES']:
            retries += 1

        response = requests.request("POST", api_url, headers=headers, data=data)
        if response.status_code == int(self.config['HTTP_SERVICE_UNAVAILABLE']):
            self.logger.info(f"Status code: {response.status_code}.")
            self.logger.info("Retrying..")
            time.sleep(1)
        else:
            return json.loads(response.content.decode("utf-8"))

        self.logger.info(f"Status code: {response.status_code}.")
        self.logger.info("Connection to the server failed after reaching maximum retry attempts.")
        self.logger.debug(f"Response: {json.loads(response.content.decode('utf-8'))}.")
        raise HTTPServiceUnavailableException("The HTTP service is unavailable.")

    def _query_in_list(self, inputs: List[Text], model: Optional[Text] = None, task: Optional[Text] = None) -> List[Union[Dict, List]]:
        return [self._query(input, model, task) for input in inputs]

    def _query_in_df(self, df, input_column: Text, model: Optional[Text] = None, task: Optional[Text] = None) -> List[Union[Dict, List]]:
        return self._query_in_list(df[input_column].tolist(), model, task)