import pandas as pd
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.pyll.base import scope

class XGBoostModel:

    """
    The class XGBoostModel includes three main methods:
    fit(train_data) trains the XGBoost model on the given training data.
    predict(test_data) generates forecasts for the given test data using the trained model.
    hyperparameter_tune(train_data, val_data, num_evals) tunes the hyperparameters of the XGBoost model using Hyperopt
         with the specified number of evaluations. It uses the mean squared error (MSE) between the true and predicted
          values as the objective function.
    In the hyperparameter_tune method, we define the search space for the hyperparameters using the hp module from Hyperopt. 
    We also define the Trials object to keep track of the results of each evaluation. 
    Then, we call the fmin function from Hyperopt to perform the hyperparameter tuning using the tpe algorithm. 
    Finally, we set the hyperparameters of the XGBoost model to the values that resulted in the lowest MSE.

    """

    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100, objective='reg:squarederror'):
        """
        Initializes an XGBoost model with the specified hyperparameters.

        Parameters:
        - max_depth: Maximum depth of the decision trees in the model.
        - learning_rate: Learning rate of the model.
        - n_estimators: Number of decision trees in the model.
        - objective: Objective function for the model. Default is 'reg:squarederror' for regression.
        """
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.objective = objective
    
    def fit(self, train_data):
        """
        Fits the XGBoost model to the training data.

        Parameters:
        - train_data: A pandas DataFrame containing the time series data to fit the model to.
                      The dataframe should contain two columns: 'ds' for dates and 'y' for values.

        Returns:
        - A trained XGBoost model.
        """
        self.trained_model = xgb.XGBRegressor(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            objective=self.objective
        )
        self.trained_model.fit(train_data[['ds']], train_data['y'])
        return self.trained_model
    
    def predict(self, test_data):
        """
        Generates forecasts for the test data using the trained XGBoost model.

        Parameters:
        - test_data: A pandas DataFrame containing the time series data to generate forecasts for.
                     The dataframe should contain one column 'ds' for dates.

        Returns:
        - A pandas DataFrame containing the forecasts.
        """
        if not hasattr(self, 'trained_model'):
            raise ValueError("Model has not been trained yet. Please call fit() before predict().")
        
        test_data = test_data[['ds']]
        test_data['yhat'] = self.trained_model.predict(test_data)
        return test_data
    
    def objective(self, params):
        model = xgb.XGBRegressor(**params)
        model.fit(train_data[['ds']], train_data['y'])
        y_pred = model.predict(val_data[['ds']])
        error = mean_squared_error(val_data['y'], y_pred)
        return error

    def hyperparameter_tune(self, train_data, val_data, num_evals=100):
        space = {
            'max_depth': scope.int(hp.quniform('max_depth', 2, 10, 1)),
            'learning_rate': hp.loguniform('learning_rate', -5, 0),
            'n_estimators': scope.int(hp.quniform('n_estimators', 50, 500, 1)),
            'objective': 'reg:squarederror'
        }
        trials = Trials()
        best = fmin(fn=self.objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=num_evals,
                    trials=trials)
        self.max_depth = int(best['max_depth'])
        self.learning_rate = best['learning_rate']
        self.n_estimators = int(best['n_estimators'])
        self.objective = 'reg:squarederror'

"""
Usage:

To use this class, you can follow these steps:

    Load the training and test data into pandas dataframes.
    Create an instance of the XGBoostModel class with default hyperparameters or specify your own hyperparameters.
    Use the hyperparameter_tune method to tune the hyperparameters of the model on a validation set.
    Use the fit method to train the model on the full training set.
    Use the predict method to generate forecasts for the test set.

# Load the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Create an instance of the XGBoostModel class
model = XGBoostModel()

# Tune the hyperparameters using the training and validation data
val_data = train_data.tail(1000)
train_data = train_data[:-1000]
model.hyperparameter_tune(train_data, val_data)

# Train the model using the full training data
model.fit(train_data)

# Generate forecasts for the test data
forecasts = model.predict(test_data)


"""