import unittest
import pandas as pd
import sys
sys.path.append('C:/Users/abdul.qadeer/Desktop/M5-Forecasting-v.2/transformer')
from LogTransformer import LogTransformer

class TestLogFeature(unittest.TestCase):
    def setUp(self):
        data = {'Price': [100, 110, 115, 120, 118],
                'Quantity': [50, 45, 60, 70, 65],
                'Revenue': [5000, 4950, 6900, 8400, 7600]}
        self.df = pd.DataFrame(data)
        
    def test_transform(self):
        lf = LogTransformer(['Price', 'Revenue'])
        transformed_df = lf.transform(self.df)
        
        # check the columns of the transformed DataFrame
        self.assertListEqual(list(transformed_df.columns), ['Price_log', 'Revenue_log', 'Quantity'])
        
        # check the values of the transformed DataFrame
        self.assertAlmostEqual(transformed_df['Price_log'][0], 4.60517019)
        self.assertAlmostEqual(transformed_df['Price_log'][1], 4.70048037)
        self.assertAlmostEqual(transformed_df['Price_log'][2], 4.74493213)
        self.assertAlmostEqual(transformed_df['Price_log'][3], 4.78749174)
        self.assertAlmostEqual(transformed_df['Price_log'][4], 4.77068462)
        
        self.assertAlmostEqual(transformed_df['Revenue_log'][0], 8.51719319)
        self.assertAlmostEqual(transformed_df['Revenue_log'][1], 8.50714338)
        self.assertAlmostEqual(transformed_df['Revenue_log'][2], 8.83973151)
        self.assertAlmostEqual(transformed_df['Revenue_log'][3], 9.0359879)
        self.assertAlmostEqual(transformed_df['Revenue_log'][4], 8.93590419)
        
        self.assertListEqual(list(transformed_df['Quantity']), [50, 45, 60, 70, 65])


if __name__ == '__main__':
    unittest.main()