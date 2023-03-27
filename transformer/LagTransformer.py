import pandas as pd

class LagTransformer:

    """
    In this class, the __init__ method initializes the class with a DataFrame of data. 
    The create_lag_feature method creates a lag feature for a specific column, 
    and returns the resulting DataFrame with the new lagged column.
    """

    def __init__(self, data):
        self.data = data
        
    def create_lag_feature(self, column, lag=1):
        """
        Create a lag feature for a specific column.
        """
        lagged_data = self.data[[column]].shift(lag)
        lagged_data.columns = [f"{column}_lag{lag}"]
        return pd.concat([self.data, lagged_data], axis=1)
    

"""
Usage: python

# create some sample data
data = {'Date': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05'],
        'Price': [100, 110, 115, 120, 118]}

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])

# create a LagFeature object with the data
lf = LagTransformer(df)

# create a lagged feature for the 'Price' column with a lag of 1
lagged_data = lf.create_lag_feature('Price', 1)

# print the results
print('Original data:')
print(df)

print('\nLagged data:')
print(lagged_data)

"""