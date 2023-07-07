from .nlp import NLP
from .computer_vision import ComputerVision
from .audio_processing import AudioProcessing

from .config_parser import ConfigParser


def get_supported_tasks():
    config_parser = ConfigParser()
    config = config_parser.get_config_dict()

    return list(config['TASK_MODEL_MAP'].keys())