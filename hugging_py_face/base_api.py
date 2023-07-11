import logging
import logging.config
from huggingface_hub import HfApi
from typing import Text, Optional

from .config_parser import ConfigParser
from .exceptions import TaskModelMismatchException

logging_config_parser = ConfigParser('config/logging.yaml')
logging.config.dictConfig(logging_config_parser.get_config_dict())
logger = logging.getLogger()


class BaseAPI:
    def __init__(self, api_token: Text, api_url: Optional[Text] = None):
        self.api_token = api_token

        config_parser = ConfigParser()
        self.config = config_parser.get_config_dict()

        if api_url:
            self.api_url = api_url
        else:
            self.api_url = self.config['BASE_URL']

        self.logger = logger

        self.hf_api = HfApi()

    def _check_model_task_match(self, model: Text, task: Text) -> None:
        metadata = self.hf_api.model_info(model)
        if task != metadata.pipeline_tag:
            raise TaskModelMismatchException(f"The task {task} is not supported by the model {model}.")

