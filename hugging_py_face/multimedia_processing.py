import json
import time
import requests
from typing import Text, Dict, List, Optional, Union

from .base_api import BaseAPI
from .exceptions import HTTPServiceUnavailableException, APICallException


class MultimediaProcessing(BaseAPI):
    def __init__(self, api_token: Text, api_url: Optional[Text] = None):
        super().__init__(api_token, api_url)

    def _query(self, input: Text, model: Optional[Text] = None, task: Optional[Text] = None) -> Union[Dict, List]:
        if model:
            self._check_model_task_match(model, task)

        api_url = f"{self.api_url}/{model if model is not None else self.config['TASK_MODEL_MAP'][task]}"

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
        elif response.status_code == 200:
            return json.loads(response.content.decode("utf-8"))
        else:
            self.logger.info(f"Status code: {response.status_code}.")
            error_message = self._extract_error_message(response)
            raise APICallException(f"API call failed with the error: {error_message}.")

        self.logger.info(f"Status code: {response.status_code}.")
        self.logger.info("Connection to the server failed after reaching maximum retry attempts.")
        self.logger.debug(f"Response: {json.loads(response.content.decode('utf-8'))}.")
        raise HTTPServiceUnavailableException("The HTTP service is unavailable.")

    def _query_in_list(self, inputs: List[Text], model: Optional[Text] = None, task: Optional[Text] = None) -> List[Union[Dict, List]]:
        return [self._query(input, model, task) for input in inputs]

    def _query_in_df(self, df, input_column: Text, model: Optional[Text] = None, task: Optional[Text] = None) -> List[Union[Dict, List]]:
        return self._query_in_list(df[input_column].tolist(), model, task)