import sys
from dataclasses import dataclass
import datetime
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.LungCancerDetection.exception import CustomException
from src.LungCancerDetection.logger import logging
import os
from sklearn.base import BaseEstimator, TransformerMixin

from src.LungCancerDetection.utils import save_object
from imblearn.over_sampling import SMOTE

class DateTransformationExtractor(BaseEstimator,TransformerMixin):  
    def __init__(self):
        pass

    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        X['diagnosis_date'] = pd.to_datetime(X['diagnosis_date'])
        X['end_treatment_date'] = pd.to_datetime(X['end_treatment_date'])

        X['treatment_duration'] = (X['end_treatment_date']-X['diagnosis_date']).dt.days
        #X.drop(columns = ['end_treatment_date','diagnosis_date'], axis=1, inplace = True)
        return X

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

          

    def get_data_transformer_object(self):
        '''
        this function is responsible for data transformation
        '''

        try:
            
            numerical_columns = ['age','bmi', 'cholesterol_level', 'treatment_duration']

            binary_columns = ['hypertension', 'asthma', 'cirrhosis', 'other_cancer']

            categorical_columns = ['gender','cancer_stage','family_history', 'smoking_status', 'treatment_type']
            
            date_columns = ['diagnosis_date', 'end_treatment_date']
            #date_columns = DateTransformationExtractor() 
            # cannot do this since ColumnTransformer only accepts string, list of column name,boolean mask in input at 3rd position

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('std scaler', StandardScaler())

            ])
            
            binary_pipeline = 'passthrough'


            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(drop='first')),
                ('scaler', StandardScaler(with_mean=False))
            ])


            # date_pipeline = Pipeline(steps=[('date transoformer', DateTransformationExtractor()),
            #     ('imputer', SimpleImputer(strategy='median')),
            #     ('std scaler', StandardScaler())
            # ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Date columns: {date_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline, numerical_columns),
                    ("binary_pipeline",binary_pipeline, binary_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)        
                ], remainder='drop'
            )

            return preprocessor
        

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("reading the tarin and test file")

            
            target_column_name = 'survived'
            date_transformer = DateTransformationExtractor()
            numerical_column = ['age','bmi', 'cholesterol_level', 'treatment_duration']

            train_df = date_transformer.fit_transform(train_df)
            test_df = date_transformer.transform(test_df)

            
            train_df.drop(columns = ['end_treatment_date','diagnosis_date'], axis=1, inplace = True)
            test_df.drop(columns = ['end_treatment_date','diagnosis_date'], axis=1, inplace = True)

            #input train dataset divided into dependent and indpeendent feature
            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_features_train_df = train_df[target_column_name]

            #input test dataset divided into dependent and indpeendent feature
            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_features_test_df = test_df[target_column_name]

            logging.info("applying the preprocessing on training and test dataframe")

            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_features_train_df)]
            test_arr = np.c_[input_features_test_arr, np.array(target_features_test_df)]

            logging.info("saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

            




        except Exception as e:
            raise CustomException(e,sys)

        
