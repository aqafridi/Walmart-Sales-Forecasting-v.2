import unittest
import pandas as pd
import sys
sys.path.append('C:/Users/abdul.qadeer/Desktop/M5-Forecasting-v.2/transformer')
from DropZeroValues import DropZeroValues

class TestDropZeroRows(unittest.TestCase):
    def setUp(self):
        data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
                'Sales': [100, 200, 0, 150, 0],
                'Region': ['North', 'South', 'East', 'West', 'South']}
        self.df = pd.DataFrame(data)
        
    def test_drop_zeros(self):
        dz = DropZeroValues(self.df)
        df_dropped = dz.drop_values(['Name','Sales','Region'])
        
        # check the number of rows and columns of the resulting DataFrame
        self.assertEqual(df_dropped.shape, (3, 3))
        
        # check that the dropped rows are correct
        expected_rows = {'Name': ['Alice', 'Bob'], 'Sales': [100, 200], 'Region': ['North', 'South']}
        expected_df = pd.DataFrame(expected_rows)
        pd.testing.assert_frame_equal(df_dropped.reset_index(drop=True), expected_df.reset_index(drop=True))

if __name__ == '__main__':
    unittest.main()