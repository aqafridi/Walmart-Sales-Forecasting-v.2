import numpy as np
class ImputeMean:


    def __init__(self,data):
        self.data = data


    def replace_zero_with_mean(self):
        """
        Replaces all zero values in the DataFrame with the mean of the non-zero values
        in the corresponding column.
        
        Args:
            self.data (pandas.DataFrame): The DataFrame to process.
            
        Returns:
            pandas.DataFrame: The processed DataFrame with zero values replaced by mean.
        """
        for col in self.data.columns:
            non_zero_values = self.data[col][self.data[col] != 0]
            if len(non_zero_values) > 0:
                mean = np.mean(non_zero_values)
                self.data[col][self.data[col] == 0] = mean
        return self.data