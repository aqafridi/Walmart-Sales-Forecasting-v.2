import modin.pandas as pd

class DataAggregator:
    """
    In this class, the __init__ method initializes the class with a DataFrame of data. 
    The `group_by` method groups the data by a specific column, and returns the resulting
    grouped data as a pandas DataFrameGroupBy object. 
    The aggregate method aggregates the data based on a specific column and function, 
    and returns the resulting aggregated data as a pandas DataFrame.

    """


    def __init__(self, data):
        self.data = data
        
    def group_by(self, column):
        """
        Group data by a specific column.
        """
        grouped_data = self.data.groupby(column)
        return grouped_data
        
    def aggregate(self, column, func):
        """
        Aggregate data based on a specific column and function.
        """
        aggregated_data = self.data.groupby(column).agg(func)
        return aggregated_data
    


"""
Usage Example:

# create some sample data
data = {'Name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35, 40, 45, 50],
        'Score': [80, 90, 85, 95, 75, 80]}

df = pd.DataFrame(data)

# create a DataAggregation object with the data
da = DataAggregator(df)

# group the data by the 'Name' column
grouped_data = da.group_by('Name')

# aggregate the data by the 'Score' column and the 'mean' function
aggregated_data = da.aggregate('Score', 'mean')

# print the results
print('Grouped data:')
print(grouped_data.head())

print('\nAggregated data:')
print(aggregated_data.head())

"""

