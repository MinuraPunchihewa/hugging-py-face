import inspect

from .nlp import NLP
from .computer_vision import ComputerVision
from .audio_processing import AudioProcessing

from .config_parser import ConfigParser


def get_supported_tasks():
    config_parser = ConfigParser()
    config = config_parser.get_config_dict()

    return list(config['TASK_MODEL_MAP'].keys())


def get_in_df_supported_tasks():
    tasks = []
    for task_family in [NLP, ComputerVision, AudioProcessing]:
        tasks += [func[0].replace('_in_df', '').replace('_', '-') for func in inspect.getmembers(task_family, predicate=inspect.isfunction) if func[0].endswith('_in_df') and 'query' not in func[0]]

    return tasks