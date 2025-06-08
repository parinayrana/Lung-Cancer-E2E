from src.LungCancerDetection.logger import logging
from src.LungCancerDetection.exception import CustomException
import sys

from src.LungCancerDetection.components.data_ingestion import DataIngestion
from src.LungCancerDetection.components.data_ingestion import DataIngestionConfig

if __name__ =='__main__':
    logging.info("the executions has started")

    try:
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)
    