from sklearn.model_selection import train_test_split

class TrainTestSplit:

    """
    In this class, the __init__ method initializes the class with a DataFrame of data, 
    the name of the target column, target_column, the size of the testing set, test_size, 
    and the random state, random_state. 
    The split_data method splits the data into training and testing sets, 
    and returns the resulting X and y training and testing sets.
    """

    def __init__(self, data, target_column, test_size=0.3, random_state=42):
        self.data = data
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        
    def split_data(self):
        """
        Split the data into training and testing sets.
        """
        X = self.data.drop(self.target_column, axis=1)
        y = self.data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test


"""
Usage Example:

# create some sample data
data = {'Price': [100, 110, 115, 120, 118],
        'Quantity': [50, 45, 60, 70, 65],
        'Revenue': [5000, 4950, 6900, 8400, 7600]}

df = pd.DataFrame(data)

# create a TrainTestSplit object with the data
tts = TrainTestSplit(df, 'Price', test_size=0.3, random_state=42)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = tts.split_data()

# print the results
print('X_train:')
print(X_train)

print('\nX_test:')
print(X_test)

print('\ny_train:')
print(y_train)

print('\ny_test:')
print(y_test)

"""