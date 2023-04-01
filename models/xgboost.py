import xgboost as xgb;
import numpy as np
import modin.pandas as pd
class XGBoostModel:

    """
    This class takes several hyperparameters (max_depth, learning_rate, n_estimators, objective)
         as arguments to initialize an XGBoost model. 
    The fit() method trains the model on the given data, and
    The predict() method generates forecasts for a given test dataset.

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
    
    def fit(self, training_data):
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
        training_data = np.asarray(training_data)
	    # split into input and output columns
        train_X, train_y = training_data[:, :-1], training_data[:, -1]
        self.trained_model.fit(train_X, train_y)
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
        
        test_data = test_data['ds']
        test_data['yhat'] = self.trained_model.predict(np.asarray(test_data))
        return test_data


"""
Usage example

# create an XGBoost model
model = XGBoostModel(max_depth=3, learning_rate=0.1, n_estimators=100, objective='reg:squarederror')

# fit the model to the training data
train_data = pd.read_csv('train_data.csv')
train_data.columns = ['ds', 'y']
model.fit(train_data)

# generate forecasts for the test data
test_data = pd.read_csv('test_data.csv')
test_data.columns = ['ds']
forecast_df = model.predict(test_data)


"""