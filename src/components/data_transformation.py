import sys
import os
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def save_object(file_path, obj):
    with open(file_path, 'wb') as f:
        joblib.dump(obj, f)
        
@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            num_features = ['writing_score','reading_score']
            cat_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch',
       'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('Scaler',StandardScaler())
                ]
            )

            logging.info("Numerical column standard Scaling Completed")
            cat_pipeline = Pipeline(
                steps = [
                    ('Encoding',OneHotEncoder(drop='first')),
                    ('Scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info("Categorical column Encoding Completed")

            preprocessor = ColumnTransformer([
                ('numerical pipeline',num_pipeline,num_features),
                ("categorical pipeline",cat_pipeline,cat_features)]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read operation on train and test data completed")

            preprocessor_obj = self.get_data_transformer_object()
            target_column_name = "math_score"

            X_train = train_df.drop([target_column_name],axis = 1)
            X_test = test_df.drop([target_column_name],axis = 1)
            y_train = train_df[target_column_name]
            y_test = test_df[target_column_name]

            logging.info("Applying preprocessing on test and train data")

            X_train_scaled = preprocessor_obj.fit_transform(X_train)
            X_test_scaled = preprocessor_obj.transform(X_test)

            logging.info('Saving preprocessing object')
            save_object(file_path = self.data_transformation_config.preprocessor_ob_file_path,obj = preprocessor_obj)
            return (
                X_train_scaled,y_train,
                X_test_scaled,y_test,
                self.data_transformation_config.preprocessor_ob_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj = DataTransformation()
    obj.initiate_data_transformation('artifacts/train.csv','artifacts/test.csv')

    