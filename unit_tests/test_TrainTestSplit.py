import unittest
import pandas as pd
import sys
sys.path.append('C:/Users/abdul.qadeer/Desktop/M5-Forecasting-v.2/transformer')
from TrainTestSplit import TrainTestSplit

class TestTrainTestSplit(unittest.TestCase):
    def setUp(self):
        data = {'Price': [100, 110, 115, 120, 118],
                'Quantity': [50, 45, 60, 70, 65],
                'Revenue': [5000, 4950, 6900, 8400, 7600]}
        self.df = pd.DataFrame(data)
        
    def test_split_data(self):
        tts = TrainTestSplit(self.df, 'Price', test_size=0.3, random_state=42)
        X_train, X_test, y_train, y_test = tts.split_data()
        
        # check the size of the training and testing sets
        self.assertEqual(len(X_train), 3)
        self.assertEqual(len(X_test), 2)
        self.assertEqual(len(y_train), 3)
        self.assertEqual(len(y_test), 2)
        
        # check the column names of the training and testing sets
        self.assertListEqual(list(X_train.columns), ['Quantity', 'Revenue'])
        self.assertListEqual(list(X_test.columns), ['Quantity', 'Revenue'])
        
        # check the values of the training and testing sets
        self.assertListEqual(list(X_train['Quantity']), [45, 70, 50])
        self.assertListEqual(list(X_test['Quantity']), [60, 65])
        self.assertListEqual(list(y_train), [110, 120, 100])
        self.assertListEqual(list(y_test), [115, 118])

if __name__ == '__main__':
    unittest.main()
"""
python test_TrainTestSplit.py

"""