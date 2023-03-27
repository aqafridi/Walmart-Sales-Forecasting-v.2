import unittest
import pandas as pd
from my_module import DataAggregation

class TestDataAggregation(unittest.TestCase):
    def setUp(self):
        data = {'Name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob', 'Charlie'],
                'Sales': [100, 200, 300, 150, 250, 350],
                'Region': ['North', 'South', 'East', 'North', 'South', 'East']}
        self.df = pd.DataFrame(data)
        
    def test_aggregate(self):
        da = DataAggregation('Name')
        agg_df = da.aggregate(self.df)
        
        # check the columns of the aggregated DataFrame
        self.assertListEqual(list(agg_df.columns), ['Name', 'Total Sales', 'Avg Sales'])
        
        # check the values of the aggregated DataFrame
        self.assertListEqual(list(agg_df['Name']), ['Alice', 'Bob', 'Charlie'])
        self.assertListEqual(list(agg_df['Total Sales']), [250, 450, 650])
        self.assertListEqual(list(agg_df['Avg Sales']), [125, 225, 325])
        
    def test_aggregate_with_groupby(self):
        da = DataAggregation(['Region', 'Name'])
        agg_df = da.aggregate(self.df)
        
        # check the columns of the aggregated DataFrame
        self.assertListEqual(list(agg_df.columns), ['Region', 'Name', 'Total Sales', 'Avg Sales'])
        
        # check the values of the aggregated DataFrame
        self.assertListEqual(list(agg_df['Region']), ['East', 'East', 'North', 'North', 'South', 'South'])
        self.assertListEqual(list(agg_df['Name']), ['Charlie', 'Bob', 'Alice', 'Charlie', 'Bob', 'Alice'])
        self.assertListEqual(list(agg_df['Total Sales']), [300, 200, 250, 350, 250, 150])
        self.assertListEqual(list(agg_df['Avg Sales']), [300, 200, 175, 350, 250, 150])


if __name__ == '__main__':
    unittest.main()