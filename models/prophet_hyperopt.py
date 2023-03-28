import pandas as pd
import numpy as np
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error
from hyperopt import hp, tpe, fmin
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample

class ProphetModel:


    
    def __init__(self):
        self.model = None
        self.params = {}

    def fit(self, train_data):
        self.model = Prophet(**self.params)
        self.model.fit(train_data)

    def predict(self, test_data):
        future = self.model.make_future_dataframe(periods=len(test_data))
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat']].tail(len(test_data))

    def objective(self, params, train_data, val_data):
        # Set model parameters
        self.params['growth'] = params['growth']
        self.params['changepoint_prior_scale'] = params['changepoint_prior_scale']
        self.params['seasonality_mode'] = params['seasonality_mode']

        # Fit model and make predictions on validation set
        self.fit(train_data)
        preds = self.predict(val_data)

        # Calculate mean squared error
        mse = mean_squared_error(val_data['y'], preds['yhat'])
        return mse

    def hyperparameter_tune(self, train_data, val_data, num_evals):
        # Define search space for hyperparameters
        space = {
            'growth': hp.choice('growth', ['linear', 'logistic']),
            'changepoint_prior_scale': scope.float(hp.quniform('changepoint_prior_scale', 0.01, 0.5, 0.01)),
            'seasonality_mode': hp.choice('seasonality_mode', ['additive', 'multiplicative'])
        }

        # Define objective function for hyperparameter tuning
        objective = lambda params: self.objective(params, train_data, val_data)

        # Use Tree-structured Parzen Estimator (TPE) algorithm for hyperparameter search
        best = fmin(objective, space, algo=tpe.suggest, max_evals=num_evals)

        # Update model parameters with best hyperparameters found
        self.params['growth'] = ['linear', 'logistic'][best['growth']]
        self.params['changepoint_prior_scale'] = round(best['changepoint_prior_scale'], 2)
        self.params['seasonality_mode'] = ['additive', 'multiplicative'][best['seasonality_mode']]
