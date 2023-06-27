import unittest

import logging
import logging.config

from hugging_py_face.config_parser import ConfigParser

logging_config_parser = ConfigParser('config/logging.yaml')
logging.config.dictConfig(logging_config_parser.get_config_dict())
logger = logging.getLogger()


def run_tests():
    loader = unittest.TestLoader()
    suite = loader.discover("tests")
    runner = unittest.TextTestRunner()
    result = runner.run(suite)

    total_tests = result.testsRun
    passed_tests = total_tests - len(result.failures) - len(result.errors)
    pass_percentage = (passed_tests / total_tests) * 100

    return pass_percentage


if __name__ == "__main__":
    pass_percentage = run_tests()

    pass_threshold = 75

    if pass_percentage < pass_threshold:
        logger.error(f"Test pass percentage ({pass_percentage:.2f}%) is below the threshold ({pass_threshold}%).")
        exit(1)
