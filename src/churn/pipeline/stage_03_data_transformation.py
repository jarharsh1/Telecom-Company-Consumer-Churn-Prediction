"""DVC stage: data transformation."""

from churn.components.data_transformation import DataTransformation
from churn.config.configuration import ConfigurationManager
from churn.logger import logger

STAGE = "Data Transformation"


def run() -> None:
    config = ConfigurationManager().get_data_transformation_config()
    DataTransformation(config).transform()


if __name__ == "__main__":
    logger.info(">>> %s stage started <<<", STAGE)
    run()
    logger.info(">>> %s stage completed <<<\n", STAGE)
