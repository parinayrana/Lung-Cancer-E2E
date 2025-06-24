import pickle
import numpy as np
import os
import sys


from src.LungCancerDetection.exception import CustomException
from src.LungCancerDetection.logger import logging
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train,y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train,y_train)

            best_model = gs.best_estimator_

            clf = CalibratedClassifierCV(base_estimator=best_model,cv=5, method="isotonic" )
            clf.fit(X_train,y_train)

            #y_train_pred = clf.predict_proba(X_train)[:, 1]

            y_test_pred = clf.predict_proba(X_test)[:, 1]

            # model.set_params(**gs.best_params_)
            # model.fit(X_train,y_train)

            # y_train_pred = model.predict(X_train)

            # y_test_pred = model.predict(X_test)

            #roc_train = roc_auc_score(y_train,y_train_pred)

            #f1_train = f1_score(y_train,y_train_pred)

            #f1_test = f1_score(y_test, y_test_pred)
            roc_test = roc_auc_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = roc_test

        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    


            
    