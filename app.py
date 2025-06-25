from src.LungCancerDetection.logger import logging
from src.LungCancerDetection.exception import CustomException
import sys

from src.LungCancerDetection.components.data_ingestion import DataIngestion
from src.LungCancerDetection.components.data_ingestion import DataIngestionConfig

from src.LungCancerDetection.components.data_transformation import DataTransformationConfig
from src.LungCancerDetection.components.data_transformation import DataTransformation
from src.LungCancerDetection.components.data_transformation import DateTransformationExtractor
from src.LungCancerDetection.components.Survival_model_trainer import ModelTrainer,ModelTrainerConfig


if __name__ =='__main__':
    logging.info("the executions has started")

    try:
        data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        data_transform_config = DataTransformationConfig()
        data_transform = DataTransformation()
        train_arr, test_arr , _ = data_transform.initiate_data_transformation(train_data_path, test_data_path)

        #model_trainer_config = ModelTrainerConfig()
        model_training = ModelTrainer()
        print(model_training.initiate_model_trainer(train_arr,test_arr))

        

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)
    