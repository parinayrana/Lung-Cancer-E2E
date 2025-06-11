from src.LungCancerDetection.logger import logging
from src.LungCancerDetection.exception import CustomException
import sys

from src.LungCancerDetection.components.data_ingestion import DataIngestion
from src.LungCancerDetection.components.data_ingestion import DataIngestionConfig

from src.LungCancerDetection.components.data_transformation import DataTransformationConfig
from src.LungCancerDetection.components.data_transformation import DataTransformation
from src.LungCancerDetection.components.data_transformation import DateTransformationExtractor

if __name__ =='__main__':
    logging.info("the executions has started")

    try:
        data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        data_transform_config = DataTransformationConfig()
        data_transform = DataTransformation()
        data_transform.initiate_data_transformation(train_data_path, test_data_path)

        

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)
    