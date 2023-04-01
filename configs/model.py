config = {

    "xgb":{
    'seed': 42,
    'max_depth':3,
    'learning_rate':0.1,
    'n_estimators':100, 
    'objective':'reg:squarederror'
    },


    "srmx":{
    'order': (1, 0, 0), 
    'seasonal_order':(0, 0, 0, 0)
    },
    

    "prpt":{
    'yearly_seasonality':True, 
    'weekly_seasonality':True, 
    'daily_seasonality':True, 
    'seasonality_mode':'additive'
    }

}
