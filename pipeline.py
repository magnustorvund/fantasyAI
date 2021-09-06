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
    
from lightgbm.sklearn import LGBMModel, LGBMRegressor
from numpy.core.numeric import full
from sklearn import model_selection
from data import create_training_data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss

import numpy as np
import pandas as pd
import lightgbm as lgb

import optuna
from optuna.integration import LightGBMPruningCallback

    
#def run_data_prep(season: str):
          

def run_model_training_quick(season: str):
    
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
    mse_test = mean_squared_error(y_test, y_pred) ** 0.5
    print(f'The RMSE of prediction is: {mse_test}')
    
    return model

def objective(trial: optuna.Trial, X, y):
    
    param_grid = {
        #         "device_type": trial.suggest_categorical("device_type", ['gpu']),
        "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05), # adjust this for speed increase
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "max_bin": trial.suggest_int("max_bin", 200, 300),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.2, 0.9, step=0.1
        ),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.2, 0.9, step=0.1
        ),

    }

    cv = KFold(n_splits=5, shuffle=True, random_state=1121218)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        # Define DV and IVs:
        X_train_full, X_test_full = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Drop the name column to avoid using it as a feature
        X_train = X_train_full.drop(X_train_full.columns[0], axis=1)
        X_test = X_test_full.drop(X_test_full.columns[0], axis=1)
    

        model = LGBMRegressor(**param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="l2", # Using Mean Squared Error as a metric (l2 in light GBM terms)
            early_stopping_rounds=100,
            callbacks=[
                LightGBMPruningCallback(trial, "l2")
            ],  # Add a pruning callback
        )

        preds = model.predict(X_test)
        cv_scores[idx] = mean_squared_error(y_test, preds)

        print(f"iteration{idx}")
    
    print(f"for loop finished!")

    return np.mean(cv_scores)


def run_hyperparameter_tuning(df: pd.DataFrame):
    # df: create_training_set() output
    
    # 0. Load and prepare training data
    X = df.drop(["event_points"], axis = 1)
    y = df["event_points"]
    
    # 1. Hyperparameter tuning
    study = optuna.create_study(direction="minimize", study_name="LGBM Regressor")
    func = lambda trial: objective(trial, X, y)
    study.optimize(func, n_trials=20)
    
    print("Study objective finished!")
    
    print(f"\tBest value (mse): {study.best_value:.5f}")
    print(f"\tBest params:")

    for key, value in study.best_params.items():
        print(f"\t\t{key}: {value}")
        
    best_hyperparams = study.best_params
    
    return best_hyperparams


def run_model_training(best_hyperparams, df):
    
    split = 0.3
        
    full_df = df
        
    train_df, test_df = train_test_split(full_df, test_size=split)
    
    # Define DV and IVs:
    y_train = train_df["event_points"]
    y_test = test_df["event_points"]
    X_train = train_df.drop(["name","event_points"], axis = 1)
    X_test = test_df.drop(["name","event_points"], axis = 1)
    
    # Create LGB dataset:
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
    model = lgb.train(best_hyperparams,
                    lgb_train,
                    num_boost_round=100, # adjust this for speed increase
                    valid_sets=lgb_test)
                    
                
                    

    print('Saving model...')
    # save model to file
    model.save_model('model.txt')

    print('Starting predicting...')
    # predict
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    # eval
    mse_test = mean_squared_error(y_test, y_pred) ** 0.5
    print(f'The MSE of prediction is: {mse_test}')
    
    return model, mse_test