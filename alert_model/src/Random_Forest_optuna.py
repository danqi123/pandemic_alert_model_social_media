# Scripts for tuning hyperparameters for Random Forest.

from startup import GOOGLE_FORECASTING_DATA, COMBINED_FORECASTING_DATA
import pandas as pd
import os
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sktime.forecasting.model_selection import SlidingWindowSplitter
import optuna

import warnings
warnings.filterwarnings("ignore")

def base_parser():

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--GPU', type=str, default='0',
                        help='Set -1 for CPU running')
    parser.add_argument('--seed', type=int,
                        default=2)
    parser.add_argument('--save_path', type=str,
                        default='../models/Random_Forest/')
    parser.add_argument('--exp_name', type=str,
                        default='Random Forest model')
    parser.add_argument('--proxy', type=str,
                        default='Combined',
                        choices=['Google', 'Combined'])
    parser.add_argument('--gold_standard', type=str,
                        default='RKI_case',
                        choices=['RKI_hospitalization', 'RKI_case'])
    parser.add_argument('--forecasting_horizon', type=int,
                        default=14)
    parser.add_argument('--training_length', type=int,
                        default=28)
    parser.add_argument('--mode', type=str,
                        default='train', choices=['train', 'test'])
    parser.add_argument('--from_best', type=bool,
                        default=False)
    parser.add_argument('--n_estimators', type=int,
                        default=300)
    parser.add_argument('--n_features', type=int,
                        default=10)
    parser.add_argument('--n_classes', type=int,
                        default=3)
    parser.add_argument('--max_depth', type=int,
                        default=75)
    parser.add_argument('--min_samples_split', type=int,
                        default=2)
    parser.add_argument('--min_samples_leaf', type=int,
                        default=1)
    parser.add_argument('--max_features', type=str,
                        default='sqrt')
    parser.add_argument('--cv_initial_window', type=int,
                        default=90)
    parser.add_argument('--cv_step_length', type=int,
                        default=70)
    parser.add_argument('--cv_test_window', type=int,
                        default=30)
    parser.add_argument('--number_trial', type=int,
                        default=90)
    parser.add_argument('--save_freq', type=int,
                        default=5)
    config = parser.parse_args()
    return config

def load_data(config):

    if config.proxy == 'Google':
        folder = GOOGLE_FORECASTING_DATA
    elif config.proxy == 'Combined':
        folder = COMBINED_FORECASTING_DATA

    train_feature =  pd.read_csv(f'{folder}/RF_data/RF_{config.proxy}_{config.gold_standard}_train_feature.csv', index_col=[0])
    test_feature =  pd.read_csv(f'{folder}/RF_data/RF_{config.proxy}_{config.gold_standard}_test_feature.csv', index_col=[0])

    train_label = pd.read_csv(f'{folder}/RF_data/RF_{config.proxy}_{config.gold_standard}_train_label.csv', index_col=[0])
    test_label = pd.read_csv(f'{folder}/RF_data/RF_{config.proxy}_{config.gold_standard}_test_label.csv', index_col=[0])

    return train_feature, train_label, test_feature, test_label

def timeseries_cv(config, training_set:pd.DataFrame, forecasting_horizon: list) -> list:
    """_summary_

    Args:
        training_set (pd.DataFrame): _description_
        initial_window (int): _description_
        forecasting_horizon (list): _description_
        step_size (int): _description_

    Returns:
        list: _description_
    """
    cv = SlidingWindowSplitter(window_length=config.cv_initial_window, fh=forecasting_horizon, step_length=config.cv_step_length)
    index = training_set.index
    timeseries_cv = []
    for train_idx, val_idx in cv.split(index):
        timeseries_cv.append((list(train_idx), list(val_idx)))
    return timeseries_cv


def objective(trial, config):
    """ objective function"""
    
    # main hyperparameters
    config.n_estimators = trial.suggest_categorical('n_estimators', [200, 300, 400, 500, 600])
    config.min_samples_split = trial.suggest_categorical('min_samples_split', [2, 5, 10])
    config.max_depth = trial.suggest_int('max_depth', 5, 50, step=10)
    config.min_samples_leaf = trial.suggest_categorical('min_samples_leaf', [1, 2, 4])
    config.max_features = trial.suggest_categorical('max_features', ['log2', 'sqrt'])
    
    # define Random Forest classifier
    RF = RandomForestClassifier(n_estimators=config.n_estimators,
                                max_depth=config.max_depth,
                                min_samples_split=config.min_samples_split,
                                min_samples_leaf=config.min_samples_leaf,
                                max_features=config.max_features)
    # load dataset
    train_sliding_feature, train_sliding_label, _, _ = load_data(config)
    

    # define time series cross validation
    forecasting_horizon = list(range(config.training_length + config.forecasting_horizon - 1, config.training_length + config.forecasting_horizon - 1 + config.cv_test_window))
    cv = timeseries_cv(config, train_sliding_feature, forecasting_horizon=forecasting_horizon)
    print(len(cv))
    # define scores: maximize accuracy
    scores = model_selection.cross_val_score(RF, train_sliding_feature, train_sliding_label.values.ravel(),
                                           n_jobs=-1, cv=cv, scoring='accuracy')
    
    return np.mean(scores)


def max_trial_callback(study, trial):
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE or t.state == optuna.trial.TrialState.RUNNING])
    if n_complete > config.number_trial - 1:
        study.stop()


if __name__ == '__main__':

    config = base_parser()
    if config.GPU != '-1':
        config.GPU_print = [int(config.GPU.split(',')[0])]
        os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU
        config.GPU = [int(i) for i in range(len(config.GPU.split(',')))]
    else:
        config.GPU = False

    np.random.seed(config.seed)

    config.save_path = os.path.join(config.save_path, config.proxy, config.gold_standard, f'RF_models_forecast_{config.forecasting_horizon}days')
    config.save_path_reports = os.path.join(config.save_path, 'reports')

    os.makedirs(config.save_path, exist_ok=True)
    os.makedirs(config.save_path_reports, exist_ok=True)

    study = optuna.create_study(study_name=config.exp_name,
                                sampler=optuna.samplers.TPESampler(),
                                direction='maximize')


    study.optimize(lambda trial: objective(trial, config), n_jobs=1, callbacks=[max_trial_callback])

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    print('Avg accuracy', trial.value)

    name_csv = os.path.join(config.save_path, 'Best_hyperparameters.csv')

    print('  Params: ')
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    trials_df = study.trials_dataframe()
    trials_df.to_csv(f'{config.save_path_reports}/CV_results.csv')
    trials_df.head()

    dic = dict(trial.params)
    dic['value'] = trial.value
    df = pd.DataFrame.from_dict(data=dic, orient='index').to_csv(name_csv, header=False)


