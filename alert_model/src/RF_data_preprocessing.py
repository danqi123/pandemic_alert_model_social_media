"""Scripts used to preprocessing the data_folder for Random Forest."""

from startup import GOOGLE_TREND, TWITTER_TREND, GOOGLE_FORECASTING_DATA, GOLD_STANDARD_TREND, COMBINED_FORECASTING_DATA, GOOGLE_LSTM, COMBINED_LSTM, GOOGLE_TRENDS_DATA, TWITTER_DATA
import pandas as pd
import os
import logging
import json
import numpy as np
import argparse
import datetime
from log_linear_regression import transfer_date
from sktime.forecasting.model_selection import SlidingWindowSplitter
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def base_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--GPU', type=str, default='0',
                        help='Set -1 for CPU running')
    parser.add_argument('--proxy', type=str,
                        default='Google',
                        choices=['Google', 'Combined'])
    parser.add_argument('--gold_standard', type=str,
                        default='RKI_hospitalization',
                        choices=['RKI_case', 'RKI_hospitalization'])
    parser.add_argument('--time_start', type=str,
                        default='2020-03-01')
    parser.add_argument('--time_end', type=str,
                        default='2022-06-30')
    parser.add_argument('--split_date', type=str,
                        default='2022-04-15')
    parser.add_argument('--training_length', type=int,
                        default=28)
    parser.add_argument('--forecasting_horizon', type=int,
                        default=14)
    parser.add_argument('--cv_initial_window', type=int,
                        default=90)
    parser.add_argument('--cv_step_length', type=int,
                        default=70)
    parser.add_argument('--cv_test_window', type=int,
                        default=30)                        
    parser.add_argument('--save_freq', type=int,
                        default=5)
    config = parser.parse_args()
    return config

def get_trend_dataset(config) -> pd.DataFrame:
    """

    Parameters
    ----------
    config

    Returns
    -------

    """
    
    if config.proxy == "Google":
        f = open(f'{GOOGLE_TRENDS_DATA}/google_top_20_symptom_and_synonyms.json')
        symptom_list = json.load(f)

        dataset = pd.DataFrame(columns=['date'])
        for sym in symptom_list["symptom_synonyms"]:
            try:
                data_file = f'{GOOGLE_TREND}/Google_Trends_{sym}_trend_label.csv'
                read_file = pd.read_csv(data_file)
                if dataset['date'].empty:
                    dataset['date'] = read_file['date']
                dataset[sym] = read_file['slope']
            except:
                continue
        dataset = dataset.set_index('date')

    elif config.proxy == "Combined":
        # get dataset for Google Trends
        pd1 = pd.read_csv(f'{GOOGLE_LSTM}/Google_{config.gold_standard}.csv', index_col='date')
        # get dataset for Twitter
        f_twitter = open(f'{TWITTER_DATA}/twitter_top_20_symptom_and_synonyms.json')
        symptom_list_twitter = json.load(f_twitter)
        pd2 = pd.DataFrame(columns=['date'])
        for sym in symptom_list_twitter["symptom_synonyms"]:
            try:
                data_file = f'{TWITTER_TREND}/Twitter_{sym}_trend_label.csv'
                read_file = pd.read_csv(data_file)
                if pd2['date'].empty:
                    pd2['date'] = read_file['date']
                pd2[sym] = read_file['slope']
            except:
                continue
        pd2 = pd2.set_index('date')
        dataset = pd.concat([pd1, pd2], axis=1)
    
    dataset = dataset[config.time_start: config.time_end]
    dataset = dataset.loc[:, (dataset != 0).any(axis=0)]

    if config.proxy == 'Google':
        dataset.to_csv(f'{GOOGLE_LSTM}/Google_{config.gold_standard}.csv')
    elif config.proxy == 'Combined':
        dataset.to_csv(f'{COMBINED_LSTM}/Combined_{config.gold_standard}.csv')
    return dataset

def create_dataset(X: pd.DataFrame, time_steps: int) -> np:
    """prepare sliding window dataset for trend forecasting. i.e. The training length is 28 days here.

    Args:
        X (pd.DataFrame): the time series dataset with slope coefficients data_folder from log-linear regression model.
        time_steps (int): we set it as 28 days (4 weeks)

    Returns:
        np: the sliding window datasets.
    """
    Xs = []
    # for i in range(len(X) - time_steps):
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].to_numpy().flatten()
        Xs.append(v)
    return np.array(Xs)

# take the last label in a sliding window as the corresponding window
def create_label(Y: pd.DataFrame, time_steps: int, forecasting_horizon: int) -> np:
    """generate sliding label of certain date (the last label of each sliding window)

    Args:
        Y (pd.DataFrame): the label dataset
        time_steps (int): training length with 28 days.
        forecasting_horizon (int)： 14 days.

    Returns:
        numpy.array: the sliding label
    """
    Ys = []
    for i in range(time_steps + forecasting_horizon - 1, len(Y)):
        v = Y[i:][0]
        Ys.append(v)
    return np.array(Ys)

def sliding_window(config, dataset: pd.DataFrame) -> tuple:
    """ sliding window approach to preprocess the dataset.

    Args:
        dataset (pd.DataFrame): the slope coefficients of digital data_folder trace

    Returns:
        tuple: sliding window feature space and label
    """
    feature = dataset[config.time_start:]
    feature_window = create_dataset(feature, config.training_length)

    RKI_df = pd.read_csv(f'{GOLD_STANDARD_TREND}/{config.gold_standard}_trend_label.csv')
    RKI_df = RKI_df.set_index("date")

    label = list(RKI_df.loc[:config.time_end, "up/down trend"])


    # change the classification task to binary problem
    label_window = create_label(label, config.forecasting_horizon, config.training_length)[:len(feature_window)]

    if len(feature_window) > len(label_window):
        feature_window = feature_window[:len(label_window)]

    sliding_columns = list(dataset.columns)
    sliding_index = list(feature.index)[config.training_length-1:]

    if len(sliding_index) < len(feature_window):
        feature_window = feature_window[:len(sliding_index)]
    else:
        sliding_index = sliding_index[:len(feature_window)]

    feature_column = sliding_columns * config.training_length

    sliding_feature = pd.DataFrame(feature_window, index=sliding_index, columns=feature_column)

    sliding_label = pd.DataFrame(label_window, columns=['label'], index=sliding_index)

    return sliding_feature, sliding_label

def timeseries_cv(training_set:pd.DataFrame, initial_window: int, forecasting_horizon: list, step_size: int) -> list:
    """perform time series cross-validation.

    Args:
        training_set (pd.DataFrame): _description_
        initial_window (int): 90 days
        forecasting_horizon (list): the forecasting list which states the indices of start and end of test set (30 days).
        step_size (int): 70 days

    Returns:
        list: the idices list of cross-validation set.
    """
    cv = SlidingWindowSplitter(window_length=initial_window, fh=forecasting_horizon, step_length=step_size)
    index = training_set.index
    print(f'The number of CV: {cv.get_n_splits(index)}')
    timeseries_cv = []
    for train_idx, val_idx in cv.split(index):
        timeseries_cv.append((list(train_idx), list(val_idx)))
    return timeseries_cv


def train_test_split(config, data: pd.DataFrame, split: str) -> tuple:
    """_summary_

    Args:
        config (_type_): _description_
        data (pd.DataFrame): _description_

    Returns:
        tuple: _description_
    """
    if split == "test":
        y1, m1, d1 = transfer_date(config.split_date)
    elif split == "val":
        y1, m1, d1 = transfer_date(config.val_split_date)
    d1 = datetime.datetime(y1, m1, d1)
    d2 = d1 + datetime.timedelta(days=config.forecasting_horizon -1 + config.training_length)
    next_start = str(d2)[:10]
    if split == "test":
        x_train = data[config.time_start:config.split_date]
    elif split == "val":
        x_train = data[config.time_start:config.val_split_date]
    return x_train, data[next_start:]

def get_end_index_for_training(config, sliding_feature: pd.DataFrame, split:str):
    """_summary_

    Args:
        config (_type_): _description_
        sliding_feature (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    x_train, _ = train_test_split(config, sliding_feature, split)
    forecasting_horizon = list(range(config.forecasting_horizon - 1 + config.training_length, config.forecasting_horizon - 1 + config.training_length + config.cv_test_window))
    cv = timeseries_cv(x_train, initial_window = config.cv_initial_window, forecasting_horizon=forecasting_horizon, step_size=config.cv_step_length)
    end_index = cv[-1][1][-1]
    return end_index

def get_next_date(date: str, days: int):
    """_summary_

    Args:
        date (str): _description_
        days (int): _description_

    Returns:
        _type_: _description_
    """
    y1, m1, d1 = transfer_date(date)
    d1 = datetime.datetime(y1, m1, d1)
    d2 = d1 + datetime.timedelta(days=days)
    next_start = str(d2)[:10]
    return next_start

def split_train_test(config, end_index: str, sliding_feature: pd.DataFrame, sliding_label: pd.DataFrame):
    """_summary_

    Args:
        config (_type_): _description_
        end_index (str): _description_
        sliding_feature (pd.DataFrame): _description_
        sliding_label (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """

    training_sliding_feature = sliding_feature.iloc[:end_index + 1, :]
    training_sliding_label = sliding_label.iloc[:end_index + 1, :]
    training_end_date = list(sliding_feature.index)[end_index]
    test_start_date = get_next_date(training_end_date, config.forecasting_horizon - 1 + config.training_length)
    test_sliding_feature = sliding_feature[test_start_date:]
    test_sliding_label = sliding_label[test_start_date:]
    return training_sliding_feature, training_sliding_label, test_sliding_feature, test_sliding_label

def compile(config):
    """_summary_

    Args:
        config (_type_): _description_

    Returns:
        _type_: _description_
    """
    dataset = get_trend_dataset(config)
    feature, label = sliding_window(config, dataset)
    end_date_index = get_end_index_for_training(config, feature, "test")
    X_train, y_train, X_test, y_test = split_train_test(config, end_date_index, feature, label)

    if config.proxy == 'Google':
        RF_data_folder = os.path.join(GOOGLE_FORECASTING_DATA, 'RF_data')
    elif config.proxy == 'Combined':
        RF_data_folder = os.path.join(COMBINED_FORECASTING_DATA, 'RF_data')

    os.makedirs(RF_data_folder, exist_ok=True)
    X_train.to_csv(f'{RF_data_folder}/RF_{config.proxy}_{config.gold_standard}_train_feature.csv')
    y_train.to_csv(f'{RF_data_folder}/RF_{config.proxy}_{config.gold_standard}_train_label.csv')
    X_test.to_csv(f'{RF_data_folder}/RF_{config.proxy}_{config.gold_standard}_test_feature.csv')
    y_test.to_csv(f'{RF_data_folder}/RF_{config.proxy}_{config.gold_standard}_test_label.csv')

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    config = base_parser()

    if config.GPU != '-1':
        config.GPU_print = [int(config.GPU.split(',')[0])]
        os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU
        config.GPU = [int(i) for i in range(len(config.GPU.split(',')))]
    else:
        config.GPU = False
    
    compile(config)

