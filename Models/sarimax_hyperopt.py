import pandas as pd
import statsmodels.api as sm
from hyperopt import fmin, hp, tpe, Trials
from hyperopt.pyll.base import scope

class SARIMAXModel:
    def __init__(self, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0)):
        self.order = order
        self.seasonal_order = seasonal_order
    
    def fit(self, train_data):
        self.trained_model = sm.tsa.statespace.SARIMAX(
            train_data['y'],
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit()
        return self.trained_model
    
    def predict(self, test_data):
        if not hasattr(self, 'trained_model'):
            raise ValueError("Model has not been trained yet. Please call fit() before predict().")
        
        pred = self.trained_model.get_prediction(start=test_data.index[0], end=test_data.index[-1], dynamic=False)
        pred_mean = pred.predicted_mean
        return pd.DataFrame({'ds': pred_mean.index, 'yhat': pred_mean.values})
    
    def objective(self, params):
        order = (params['p'], params['d'], params['q'])
        seasonal_order = (params['P'], params['D'], params['Q'], params['m'])
        model = sm.tsa.statespace.SARIMAX(
            train_data['y'],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit()
        y_pred = model.get_prediction(start=val_data.index[0], end=val_data.index[-1], dynamic=False).predicted_mean
        error = mean_squared_error(val_data['y'], y_pred)
        return error

    def hyperparameter_tune(self, train_data, val_data, num_evals=100):
        space = {
            'p': scope.int(hp.quniform('p', 0, 5, 1)),
            'd': scope.int(hp.quniform('d', 0, 2, 1)),
            'q': scope.int(hp.quniform('q', 0, 5, 1)),
            'P': scope.int(hp.quniform('P', 0, 5, 1)),
            'D': scope.int(hp.quniform('D', 0, 2, 1)),
            'Q': scope.int(hp.quniform('Q', 0, 5, 1)),
            'm': scope.int(hp.quniform('m', 1, 12, 1))
        }
        trials = Trials()
        best = fmin(fn=self.objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=num_evals,
                    trials=trials)
        self.order = (int(best['p']), int(best['d']), int(best['q']))
        self.seasonal_order = (int(best['P']), int(best['D']), int(best['Q']), int(best['m']))



"""
Usage example:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from SARIMAXModel import SARIMAXModel

# Load data
df = pd.read_csv('data.csv')
train_data, test_data = train_test_split(df, test_size=0.2, shuffle=False)

# Create SARIMAX model
model = SARIMAXModel()

# Hyperparameter tuning
val_data = train_data.tail(12) # Use last 12 months as validation set
model.hyperparameter_tune(train_data, val_data, num_evals=50)

# Fit model on entire training set
model.fit(train_data)

# Make predictions on test set
preds = model.predict(test_data)

# Evaluate model performance
mse = mean_squared_error(test_data['y'], preds['yhat'])
print("MSE:", mse)



"""