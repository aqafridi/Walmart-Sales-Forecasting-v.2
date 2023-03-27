import pandas as pd

class DropZeroValues:

    """
    In this class, the __init__ method initializes the class with a DataFrame of data. 
    The drop_values method drops rows with zero values in specified columns, and 
    returns the resulting DataFrame.
    
    """
    def __init__(self, data):
        self.data = data
        
    def drop_values(self, columns):
        """
        Drop rows with zero values in specified columns.
        """
        non_zero_data = self.data.loc[self.data[list(columns)].ne(0).all(axis=1)]
        return non_zero_data
    
"""
usage: python

# create some sample data
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 0, 35, 45],
        'Score': [80, 90, 0, 85]}

df = pd.DataFrame(data)

# create a DropZeroRows object with the data
dzr = DropZeroValues(df)

# drop rows with zero values in the 'Age' and 'Score' columns
non_zero_data = dzr.drop_values(['Age', 'Score'])

# print the results
print('Original data:')
print(df)

print('\nNon-zero data:')
print(non_zero_data)
"""