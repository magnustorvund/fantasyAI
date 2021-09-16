"""
    The goal for the prediction model is to predict as accurately as possible the players with highest points in next gameweek (regression). 
    Therefore, the model needs to consider the differences each position gets points for. 
    This can either be solved by a) one model, or b) run 1 model specifically designed for each position. 
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
    
from textwrap import indent
from lightgbm.sklearn import LGBMRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from datetime import datetime

from data import create_evaluation_data, create_training_data

import pickle
import copy
import json

import numpy as np
import pandas as pd
import lightgbm as lgb

import optuna
from optuna.integration import LightGBMPruningCallback

def get_data_for_pipeline(training_set_season = "2020-21"):
    
    """
    Get the datasets needed for running the pipeline. 
    
    Args:
        season (str): A string specifying the season to train the model on in the form of i.e. "2020-21"

    Returns:
        Two dataframes, one for training the model and one for predicting the current players next gameweek points
    """
    
    training_df = create_training_data(season=training_set_season)
    evaluation_df = create_evaluation_data()
    
    return training_df, evaluation_df


def run_model_training_quick(season: "2020-21"):
    
    """
    Run the light GBM model quick to allow for quick testing of variables etc.

    Args:
        season (str): A string specifying the season in the form of "2020-21"

    Returns:
        A dataframe containing the team id and name for the specified season
    """
    
    split = 0.2
        
    full_df = create_training_data(season)
    
    variables_to_keep = ['name','ict_index', 'bps', 'minutes', 'now_cost', 'fdr', 'event_points']
    full_df = full_df[variables_to_keep]
      
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
                    num_boost_round=5,
                    valid_sets=lgb_test,
                    early_stopping_rounds=100)

    print('Saving model...')
    # save model to file
    #model.save_model('model.txt')

    print('Starting predicting...')
    # predict
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    # eval
    mse_test = mean_squared_error(y_test, y_pred)
    print(f'The RMSE of prediction is: {mse_test}')
    
    # Saving the model
    now = datetime.now()
    time = now.strftime("%Y_%d_%m_%H_%M")
    name = "model_" + time
    filename = name + '.pkl'
    pickle.dump(model, open(filename, 'wb'))
    
    return model

def objective(trial: optuna.Trial, X, y):
    
    """
    The objective function for tuning the Light GBM hyperparameters with Optuna.

    Args:
        trial (optuna.Trial): An optuna trial object from the study.optimize function.
        X: The features in the light GBM model
        y: The dependent variable

    Returns:
        MSE scores from the 5-fold cross validation
    """
    
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
    
    """
    A function for creating the optuna objects and run the hyperparameter tuning

    Args:
        df (pd.DataFrame): The training dataframe from get_data_for_pipeline()

    Returns:
        A dictionary containing the best hyperparameters from the hyperparameter tuning
    """
    
    # df: create_training_set() output
    variables_to_keep = ['name','ict_index', 'bps', 'minutes', 'now_cost', 'fdr', 'event_points']
    df = df[variables_to_keep]
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
    
    print('Saving optimal parameteres...')
    now = datetime.now()
    time = now.strftime("%Y_%d_%m_%H_%M")
    name = "parameters_" + time
    filename = name + '.pkl'
    pickle.dump(best_hyperparams, open(filename, 'wb'))
    
    return best_hyperparams


def run_model_training(best_hyperparams, df):
    
    """
    Run the Light GBM model training

    Args:
        best_hyperparams: A dictionary containing hyperparameters for Light GBM
        df: The training dataframe from get_data_for_pipeline()

    Returns:
        The model and test MSE scores
    """
    best_hyperparams["metric"] = "l2"
    
    split = 0.2
    
    variables_to_keep = ['ict_index', 'bps', 'minutes', 'now_cost', "fdr", "event_points"]
    full_df = df[variables_to_keep]
        
    train_df, test_df = train_test_split(full_df, test_size=split)
    
    # Define DV and IVs:
    y_train = train_df["event_points"]
    y_test = test_df["event_points"]
    X_train = train_df.drop(["event_points"], axis = 1)
    X_test = test_df.drop(["event_points"], axis = 1)
    
    # Create LGB dataset:
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
    # Dictionary for train MSE results
    train_results = {}
    
    model = lgb.train(best_hyperparams,
                    lgb_train,
                    num_boost_round=20000, # adjust this for speed increase
                    valid_sets=[lgb_train, lgb_test],
                    early_stopping_rounds=5000,
                    evals_result=train_results
                    )
                    
    print('Saving model...')
    now = datetime.now()
    time = now.strftime("%Y_%d_%m_%H_%M")
    name = "model_" + time
    filename = name + '.pkl'
    pickle.dump(model, open(filename, 'wb'))
    
    print('Starting predicting...')
    # predict
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    # eval
    mse_test = mean_squared_error(y_test, y_pred)
    print(f'The MSE of prediction is: {mse_test}')
    mse_test_dict = {'test_mse': mse_test}
    
    with open('mse_test_dict.json', 'w', encoding='utf-8') as f:
        json.dump(mse_test_dict, f, ensure_ascii=False, indent=4)
    
    with open('train_results.json', 'w', encoding='utf-8') as f:
        json.dump(train_results, f, ensure_ascii=False, indent=4)
    
    return model, mse_test, train_results

def run_predictions(pred_df: pd.DataFrame, model_name: str):
    
    """
    Run the predictions for the next gameweek

    Args:
        pred_df (pd.DataFrame): The evaluation dataframe with players you want predictions for
        model_name (str): The name of the trained model ("model_year_month_hour_minute")

    Returns:
        A dataframe containing predictions for all current premier league players for next gameweek
    """
    
    model = pickle.load(open(model_name,'rb'))
    
    variables_to_keep = ['name','ict_index', 'bps', 'minutes', 'now_cost', 'fdr']
    pred_df = pred_df[variables_to_keep]
    
    X_eval = pred_df.drop(["name"], axis = 1)
    
    preds = model.predict(X_eval, num_iteration=model.best_iteration)
    
    output_df = copy.copy(pred_df)
    output_df["predicted_points"] = preds
    
    
    output_df = output_df.sort_values(by=["predicted_points"], ascending=False)
    
    # saving the dataframe to CSV for later review after the next GW
    print('Saving model...')
    now = datetime.now()
    time = now.strftime("%Y_%d_%m_%H_%M")
    name = "predictions_" + time
    filename = name + '.csv'
    output_df.to_csv(filename, encoding='utf-8')
    
    return output_df



