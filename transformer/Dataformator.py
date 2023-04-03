
class DataFormator:
    """This class takes two dataframes  and formate the data.
    """
    def __init__(self, data1,data2):
        self.data1 = data1
        self.data2 = data2

    def format_data(self,col1,col2):
        """This function takes two column as an argument and returns a dataframe

        Args:
            col1 (str): column 1 of aggregation
            col2 (str): column 2 of aggregation

        Returns:
            dataFrame: formatted data
        """        
        new_col = f'{col1}_{col2}'
        self.data1[new_col] = self.data1[col1].str.cat(self.data1[col2], sep='_')
        self.data1.drop(columns=[col2,col1],axis=1, inplace=True) # Drop the unnecessary columns
        data = self.data1.T  # seting the days as index for new dataframe(Data)
        data.columns = self.data1[new_col] # ading product id column to Data 
        data.index.name = None   # remove the name of index
        data.drop(index=new_col,inplace=True) # remove the first row
        # change the format and the freq. of index to datetime format
        data.index = self.data2['date'][0:1913]
        # data.index = pd.to_datetime(data.index)
        data.index = data.index
        data.index.freq= 'd'
        return data