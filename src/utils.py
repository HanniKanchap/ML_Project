from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import RandomizedSearchCV
import sys
import joblib
from src.exception import CustomException

def load_object(path):
    try:
        with open(path, 'rb') as f:
            obj = joblib.load(f)
        return obj
    except Exception as e:
        
        raise CustomException(e, sys)


def evaluate_model(X_train,X_test,y_train,y_test,models,params):
    performance_matrix = {}
    for name,model in models.items():
        param = params[name]
        model = RandomizedSearchCV(model,param_distributions=param,cv=5)

        model.fit(X_train,y_train)
        y_pred = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        print('-'*10,name,'-'*10)
        print(' '*10,"Training Data Performance\n")
        eval_model(y_train,y_pred)

        print(' '*10,"\nTesting Data Performance\n")
        val = eval_model(y_test,y_pred_test)
        
        performance_matrix[name] = (val,model.best_estimator_)
    
    return performance_matrix

def eval_model(y_test,y_pred):
    print("MSE : ",mean_squared_error(y_test,y_pred))
    print("MAE : ",mean_absolute_error(y_test,y_pred))
    score = r2_score(y_test,y_pred)
    print("R2 Score : ",score)
    return score
