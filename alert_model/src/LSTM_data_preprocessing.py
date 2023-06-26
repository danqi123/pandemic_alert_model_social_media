# Scripts used for preprocessing data_folder for LSTMs.

import pandas as pd
from startup import GOOGLE_LSTM, GOLD_STANDARD_TREND, COMBINED_LSTM
from log_linear_regression import transfer_date
import datetime


def read_label(trend: str)->list:
    """
    Generate the labels of classification task.
    """
    RKI_df = pd.DataFrame()
    if trend == "confirmed_cases":
        RKI_df = pd.read_csv(f'{GOLD_STANDARD_TREND}/RKI_case_trend_label.csv')
    elif trend == "hospitalization":
        RKI_df = pd.read_csv(f'{GOLD_STANDARD_TREND}/RKI_hospitalization_trend_label.csv')
    elif trend == "deaths":
        RKI_df = pd.read_csv(f'{GOLD_STANDARD_TREND}/RKI_death_trend_label.csv')
    RKI_df = RKI_df.set_index("date")
    label = list(RKI_df.loc[:, "up/down trend"])

    # change -1 to 2, refers to downtrends. (1: uptrends, 0: notrends, -1: downtrends)
    for i in range(len(label)):
        if label[i] == -1:
            label[i] = 2
    return label

def create_dataset(X:pd.DataFrame, time_steps:int):
    """
    Transform raw dataset into sliding windows, and return a list of dataframes.
    """
    Xs = []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)]
        Xs.append(v)
    return Xs

def create_label(Y:list, time_steps:int, forecasting_horizon: int):
    """
    The label is the last trend (RKI confirmed cases/ RKI hospitalization) of each sliding window.
    """
    Ys = []
    for i in range(time_steps + forecasting_horizon - 1, len(Y)):
        v = Y[i:][0]
        Ys.append(v)
    return Ys

def get_date(date_: str, days: int):
    """

    """
    y1, m1, d1 = transfer_date(date_)
    d1 = datetime.datetime(y1, m1, d1)
    d2 = str(d1 - datetime.timedelta(days=days))
    d2 = d2[:10]
    return d2

def data_preprocessing_pipeline(proxy: str, trend: str, training_length: int, forecasting_horizon: int, testset_split: str):
    if proxy == 'Google':
        folder = GOOGLE_LSTM
        dataset = pd.read_csv(f'{folder}/Google_RKI_case.csv')
    elif proxy == 'Combined':
        folder = COMBINED_LSTM
        dataset = pd.read_csv(f'{folder}/Combined_RKI_case.csv')

    dataset = dataset.set_index('date')

    new_dataset = dataset["2020-03-01":]
    dataset = new_dataset.copy()
    test_dataset = dataset.copy()

    # according to Random Forest test set, we gave the same start date for LSTM test sets. "2022-04-27"
    train_dataset = dataset[dataset.index <= get_date(testset_split, training_length-1)]
    test_dataset = test_dataset[test_dataset.index >= get_date(testset_split, training_length-1)]
    train_index = train_dataset.iloc[training_length-1:-forecasting_horizon, :].index

    return train_dataset, train_index, test_dataset
