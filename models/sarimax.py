import pandas as pd
import numpy as np
import statsmodels.api as sm

class SARIMAXModel:
    """
    This class takes an order tuple (p, d, q) and an optional
    seasonal_order tuple (P, D, Q, s) as arguments to initialize a SARIMAX model. 
    The fit() method trains the model on the given data, and 
    The predict() method generates forecasts and confidence intervals for a given test dataset.

    """


    def __init__(self, order, seasonal_order=None):
        """
        Initializes a SARIMAX model with the specified order and seasonal_order.

        Parameters:
        - order: A tuple of integers (p, d, q) representing the ARIMA order.
        - seasonal_order: A tuple of integers (P, D, Q, s) representing the seasonal ARIMA order.
          If not specified, the model will be fitted without seasonal effects.
        """
        self.order = order
        self.seasonal_order = seasonal_order
    
    def fit(self, train_data):
        """
        Fits the SARIMAX model to the training data.

        Parameters:
        - train_data: A pandas DataFrame containing the time series data to fit the model to.

        Returns:
        - A trained SARIMAX model.
        """
        if self.seasonal_order is not None:
            model = sm.tsa.statespace.SARIMAX(train_data, order=self.order, seasonal_order=self.seasonal_order)
        else:
            model = sm.tsa.statespace.SARIMAX(train_data, order=self.order)
        self.trained_model = model.fit()
        return self.trained_model
    
    def predict(self, test_data):
        """
        Generates forecasts for the test data using the trained SARIMAX model.

        Parameters:
        - test_data: A pandas DataFrame containing the time series data to generate forecasts for.

        Returns:
        - A pandas DataFrame containing the forecasts and confidence intervals.
        """
        if not hasattr(self, 'trained_model'):
            raise ValueError("Model has not been trained yet. Please call fit() before predict().")
        
        forecast = self.trained_model.forecast(len(test_data))
        forecast_ci = self.trained_model.get_forecast(len(test_data)).conf_int()
        
        forecast_df = pd.DataFrame({
            'forecast': forecast,
            'lower_ci': forecast_ci.iloc[:, 0],
            'upper_ci': forecast_ci.iloc[:, 1]
        }, index=test_data.index)
        
        return forecast_df


"""
Usage example:

# create a SARIMAX model with order (1, 1, 1)
model = SARIMAXModel(order=(1, 1, 1))

# fit the model to the training data
train_data = pd.read_csv('train_data.csv', index_col=0)
model.fit(train_data)

# generate forecasts for the test data
test_data = pd.read_csv('test_data.csv', index_col=0)
forecast_df = model.predict(test_data)

"""