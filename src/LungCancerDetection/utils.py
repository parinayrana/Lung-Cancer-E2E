import pickle
import numpy as np
import os
import sys


from src.LungCancerDetection.exception import CustomException
from src.LungCancerDetection.logger import logging
import pandas as pd



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

    