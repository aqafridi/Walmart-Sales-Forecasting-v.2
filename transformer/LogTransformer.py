import modin.pandas as pd
import numpy as np

class LogTransformer:
    """
    In this class, the __init__ method initializes the class with a DataFrame of data.
    The create_log_feature method creates a log feature for a specific column, 
    and returns the resulting DataFrame with the new log column.
    """


    def __init__(self, data):
        self.data = data
        
    def create_log_feature(self, column):
        """
        Create a log feature for a specific column.
        """
        log_data = pd.DataFrame(np.log(self.data[column]), columns=[f"log_{column}"])
        return pd.concat([self.data, log_data], axis=1)
    

"""
Usage Example:

# create some sample data
data = {'Date': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05'],
        'Price': [100, 110, 115, 120, 118]}

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])

# create a LogFeature object with the data
lf = LogTransformer(df)

# create a log feature for the 'Price' column
log_data = lf.create_log_feature('Price')

# print the results
print('Original data:')
print(df)

print('\nLog data:')
print(log_data)
"""