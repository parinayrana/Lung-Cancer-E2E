import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.LungCancerDetection.exception import CustomException
from src.LungCancerDetection.logger import logging
import os


@dataclass
class DatTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DatTransformationConfig()

    def get_data_transformer_object(self):
        