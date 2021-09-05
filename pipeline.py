"""
    The goal for the prediction model is to predict as accurately as possible the points each player will get in next (1,3, 5 or 10 i.e.) gameweeks (regression)
    Therefore, the model needs to consider the differences each position gets points for. 
    This can either be solved by a) one model (hard), or b) run 1 model specifically designed for each position. 
    Ideas to features in the prediction model:
    To mitigate the underlying random nature of football an idea is to rather use expected than actual statistics (through weights, which could be estimated by another model)
    - shots on target
    - expected goals (if striker, midfielder or defender)
    - expected clean sheets (if goalkeeper, defender or midfielder)
    - expected saves (if keeper)
    - actual saves (last match, last 7 matches, average last year etc.)
    - actual clean sheets (last match, last 7 matches etc.)
    - 
"""
    
from numpy.core.numeric import full
from data import create_training_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
    
#def run_data_prep(season: str):
          

def run_model_training(season: str):
    
    split = 0.2
        
    full_df = create_training_data(season)
        
    train_df, test_df = train_test_split(full_df, test_size=split)
    
    # Define DV and IVs:
    y_train = train_df["event_points"]
    y_test = test_df["event_points"]
    X_train = train_df.drop(["name","event_points"], axis = 1)
    X_test = test_df.drop(["name","event_points"], axis = 1)
    
    # Create LGB dataset:
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
    # Insert K-fold cross validation of certain parameters
    
    # Light GBM parameter configuration
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'l1'},
        'max_depth': 7,
        'num_iterations': 100,
        'num_threads': 4,
        'learning_rate': 0.05,
        'verbose': 0
    }
    
    print('Starting training...')

    model = lgb.train(params,
                    lgb_train,
                    num_boost_round=20,
                    valid_sets=lgb_test,
                    early_stopping_rounds=100)

    print('Saving model...')
    # save model to file
    model.save_model('model.txt')

    print('Starting predicting...')
    # predict
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    # eval
    rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
    print(f'The RMSE of prediction is: {rmse_test}')
    
    return model