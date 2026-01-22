import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import save_object

from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from src.utils import evaluate_model

from src.components.data_transformation import DataTransformation

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self,X_tr,y_tr,X_te,y_te):
        try:
            logging.info("Split training and Test input data")
            X_train,y_train,X_test,y_test = X_tr,y_tr,X_te,y_te
            
            models = {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "RandomForestRegressor": RandomForestRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "SVR": SVR(),
                "CatBoostRegressor": CatBoostRegressor(verbose=0),  
                "XGBRegressor": XGBRegressor()
            }
            
            params = {
            "LinearRegression": {
                "fit_intercept": [True, False]
            },
            "Lasso": {
                "alpha": [0.01, 0.1, 1],
                "max_iter": [500, 1000]
            },
            "Ridge": {
                "alpha": [0.01, 0.1, 1],
                "solver": ["auto", "svd", "cholesky"]
            },
            "RandomForestRegressor": {
                "n_estimators": [50, 100],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt",'log2']
            },
            "AdaBoostRegressor": {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1, 0.5],
                "loss": ["linear", "square"]
            },
            "KNeighborsRegressor": {
                "n_neighbors": [3, 5, 7],
                "weights": ["uniform", "distance"],
                "p": [1, 2]
            },
            "DecisionTreeRegressor": {
                "criterion": ["squared_error", "absolute_error"],
                "splitter": ["best", "random"],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            },
            "SVR": {
                "kernel": ["linear", "rbf"],
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto"]
            },
            "CatBoostRegressor": {
                "iterations": [100, 200],   # low iterations
                "depth": [4, 6],
                "learning_rate": [0.05, 0.1],
                "l2_leaf_reg": [3, 5]
            },
            "XGBRegressor": {
                "n_estimators": [100, 200],  # low iterations
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }
}
            
            logging.info("Training models")
            model_report = evaluate_model(X_train,X_test,y_train,y_test,models,params)
            logging.info("Got model performance report")

            sorted_model_report = list(sorted(model_report.items(),key = lambda x:x[1][0],reverse = True))
            print(sorted_model_report)
            best_model = sorted_model_report[0][1][1]
            
            if sorted_model_report[0][1][0] < 0.6:
                raise CustomException("No best model found",sys)
            
            logging.info("Best model Found")
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj = best_model)


        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == '__main__':
    dt = DataTransformation()
    X_train,y_train,X_test,y_test,_ =dt.initiate_data_transformation('artifacts/train.csv','artifacts/test.csv')
    obj = ModelTrainer()
    obj.initiate_model_training(X_train,y_train,X_test,y_test)