import pandas as pd
from fbprophet import Prophet

class ProphetModel:

    """
    This class takes several seasonality parameters (yearly_seasonality, weekly_seasonality, daily_seasonality)
        and a seasonality_mode parameter as arguments to initialize a Prophet model. 
    The fit() method trains the model on the given data, and
    The predict() method generates forecasts and confidence intervals for a given test dataset.

    """


    def __init__(self, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True, seasonality_mode='additive'):
        """
        Initializes a Prophet model with the specified seasonality parameters.

        Parameters:
        - yearly_seasonality: Boolean indicating whether to include yearly seasonality in the model.
        - weekly_seasonality: Boolean indicating whether to include weekly seasonality in the model.
        - daily_seasonality: Boolean indicating whether to include daily seasonality in the model.
        - seasonality_mode: Mode of seasonality components. Either 'additive' (default) or 'multiplicative'
        """
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
    
    def fit(self, train_data):
        """
        Fits the Prophet model to the training data.

        Parameters:
        - train_data: A pandas DataFrame containing the time series data to fit the model to.
                      The dataframe should contain two columns: 'ds' for dates and 'y' for values.

        Returns:
        - A trained Prophet model.
        """
        self.trained_model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode
        )
        self.trained_model.fit(train_data)
        return self.trained_model
    
    def predict(self, test_data, include_history=True):
        """
        Generates forecasts for the test data using the trained Prophet model.

        Parameters:
        - test_data: A pandas DataFrame containing the time series data to generate forecasts for.
                     The dataframe should contain one column 'ds' for dates.

        Returns:
        - A pandas DataFrame containing the forecasts and confidence intervals.
        """
        if not hasattr(self, 'trained_model'):
            raise ValueError("Model has not been trained yet. Please call fit() before predict().")
        
        future = self.trained_model.make_future_dataframe(
            periods=len(test_data),
            include_history=include_history
        )
        forecast = self.trained_model.predict(future)
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(len(test_data))
        
        return forecast_df


"""
Usage example:

# create a Prophet model
model = ProphetModel(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)

# fit the model to the training data
train_data = pd.read_csv('train_data.csv')
train_data.columns = ['ds', 'y']
model.fit(train_data)

# generate forecasts for the test data
test_data = pd.read_csv('test_data.csv')
test_data.columns = ['ds']
forecast_df = model.predict(test_data)

"""