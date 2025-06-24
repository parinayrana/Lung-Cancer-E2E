import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from src.LungCancerDetection.exception import CustomException
from src.LungCancerDetection.logger import logging
from src.LungCancerDetection.utils import save_object,evaluate_models
from sklearn.metrics import roc_auc_score



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array, test_array):
        try:
            logging.info("split training and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Logistic Regression": LogisticRegression(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Support Vector Classifier": SVC(probability=True),
                "XGBoost Classifier": XGBClassifier(eval_metric='logloss'),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "GradientBoost Classifier": GradientBoostingClassifier()
            }

            params = {
                "Logistic Regression": {'penalty':['l2'], 'C': [0.01,0.1,1,10], 'solver':['liblinear'], 'max_iter':[100,200]},
                "K-Nearest Neighbors": {'n_neighbors':[3,5,7,9], 'weights': ['uniform', 'distance'], 'metric':['euclidean','manhattan']},
                "Decision Tree": {'criterion':['gini', 'entropy'], 'max_depth': [5,10,20,25], 'min_samples_split':[2,5,10]},
                "Random Forest": {'n_estimators': [50, 100, 200],'max_depth': [5,10,20,25],'min_samples_split': [2, 5],'criterion': ['gini', 'entropy']},
                "Support Vector Classifier": {'C':[0.1,1,10], 'kernel':['linear','rbf','poly'], 'gamma':['scale', 'auto']},
                "XGBoost Classifier": {'n_estimators': [50,100,200], 'learning_rate': [0.01,0.05,0.1], 'max_depth': [3,5,7], 'subsample':[0.8,1.0], 'colsample_bytree':[0.8, 1.0]},
                "CatBoost Classifier": {'iterations':[100,200], 'learning_rate': [0.01,0.05,0.1], 'depth':[4,6,8]},
                "AdaBoost Classifier": {'n_estimators': [50,100,200],'learning_rate':[0.01,0.1,1]},
                "GradientBoost Classifier": {'n_estimators': [50,100,200], 'learning_rate':[0.01,0.05,0.1],'max_depth':[3,5,7]}
            }
            model_report : dict=evaluate_models(X_train,y_train,X_test,y_test, models, params)
            
            #to get best model score from model_report
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.5:
                raise Exception("No best model found")
            logging.info(f"Best found model on both training and testing dataset {best_model_name}")

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            predicted = best_model.predict_proba(X_test)[:,1]

            roc_Score = roc_auc_score(y_test, predicted)

            return roc_Score

        except Exception as e:
            raise CustomException(e,sys)


