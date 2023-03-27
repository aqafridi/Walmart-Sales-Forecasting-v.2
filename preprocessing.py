from sklearn.model_selection import train_test_split

class Preprocessing():
    def __init__(self):
        print("Preprocessing called")

    def data_aggregation(self,df):
        sales_store_agg = df.groupby('store_id','dept_id').sum()
        print("Data aggregation called")

        return sales_store_agg

    def mark_zero(self,df):
        print("Marking zero called")

        return df

    def impute_mean(self,df):
        print("ImputeMean called")
        
        return df

    def train_test_split(self,data,target):
        print("Train test split called")
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)

        return X_train, X_test, y_train, y_test
