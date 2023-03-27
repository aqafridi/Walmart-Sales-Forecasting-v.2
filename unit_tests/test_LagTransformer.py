import unittest
import pandas as pd
from my_module import LagFeature

class TestLagFeature(unittest.TestCase):
    def setUp(self):
        data = {'Price': [100, 110, 115, 120, 118],
                'Quantity': [50, 45, 60, 70, 65],
                'Revenue': [5000, 4950, 6900, 8400, 7600]}
        self.df = pd.DataFrame(data)
        
    def test_transform(self):
        lf = LagFeature(['Price', 'Revenue'], lags=[1,2])
        transformed_df = lf.transform(self.df)
        
        # check the columns of the transformed DataFrame
        self.assertListEqual(list(transformed_df.columns), ['Price', 'Revenue', 'Quantity', 'Price_lag1', 'Price_lag2', 'Revenue_lag1', 'Revenue_lag2'])
        
        # check the values of the transformed DataFrame
        self.assertListEqual(list(transformed_df['Price_lag1']), [None, 100, 110, 115, 120])
        self.assertListEqual(list(transformed_df['Price_lag2']), [None, None, 100, 110, 115])
        self.assertListEqual(list(transformed_df['Revenue_lag1']), [None, 5000, 4950, 6900, 8400])
        self.assertListEqual(list(transformed_df['Revenue_lag2']), [None, None, 5000, 4950, 6900])
        
        self.assertListEqual(list(transformed_df['Quantity']), [50, 45, 60, 70, 65])
        self.assertListEqual(list(transformed_df['Price']), [100, 110, 115, 120, 118])
        self.assertListEqual(list(transformed_df['Revenue']), [5000, 4950, 6900, 8400, 7600])


if __name__ == '__main__':
    unittest.main()