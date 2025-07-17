import sys
import pandas as pd
from src.LungCancerDetection.exception import CustomException
from src.LungCancerDetection.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","Survival_model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict_proba(data_scaled)[0][1]
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class PatientData:
    def __init__(
        self,
        age: int,
        bmi: float,
        cholesterol: int,
        treatment_duration: int,
        gender: str,
        stage: str,
        family_history: str,
        smoking_status: str,
        treatment_type: str,
        hypertension: bool,
        asthma: bool,
        cirrhosis: bool,
        other_cancer: bool
    ):

        self.age = age
        self.bmi = bmi
        self.cholesterol = cholesterol
        self.treatment_duration = treatment_duration

        self.gender = gender
        self.stage = stage
        self.family_history = family_history
        self.smoking_status = smoking_status
        self.treatment_type = treatment_type

        self.hypertension = hypertension
        self.asthma = asthma
        self.cirrhosis = cirrhosis
        self.other_cancer = other_cancer


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
        'age': [self.age],
        'bmi': [self.bmi],
        'cholesterol_level': [self.cholesterol],
        'treatment_duration': [self.treatment_duration],
        'gender': [self.gender],
        'cancer_stage': [self.stage],
        'family_history': [self.family_history],
        'smoking_status': [self.smoking_status],
        'treatment_type': [self.treatment_type],
        'hypertension': [int(self.hypertension)],
        'asthma': [int(self.asthma)],
        'cirrhosis': [int(self.cirrhosis)],
        'other_cancer': [int(self.other_cancer)]
    }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
