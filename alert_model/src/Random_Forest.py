# Scripts for running Random Forest algorithm

from startup import GOOGLE_FORECASTING_DATA, COMBINED_FORECASTING_DATA, GOOGLE_TRENDS_DAILY, TWITTER_DAILY, GOOGLE_FORECASTING_REPORT_RF, COMBINED_FORECASTING_REPORT_RF
import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import classification_report

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
    parser.add_argument('--proxy', type=str,
                        default='Google',
                        choices=['Google', 'Combined'])
    parser.add_argument('--gold_standard', type=str,
                        default='RKI_hospitalization',
                        choices=['RKI_hospitalization', 'RKI_case'])
    parser.add_argument('--forecasting_horizon', type=int,
                        default=14)
    parser.add_argument('--mode', type=str,
                        default='train', choices=['train', 'test'])
    parser.add_argument('--n_estimators', type=int,
                        default=700)
    parser.add_argument('--max_depth', type=int,
                        default=15)
    parser.add_argument('--min_samples_split', type=int,
                        default=10)
    parser.add_argument('--min_samples_leaf', type=int,
                        default=1)
    parser.add_argument('--max_features', type=str,
                        default='sqrt')
    parser.add_argument('--save_freq', type=int,
                        default=5)
    config = parser.parse_args()
    return config

def get_symptom_list(config):
    if config.proxy == "Google":
        df = pd.read_csv(GOOGLE_TRENDS_DAILY)
    elif config.proxy == "Twitter":
        df = pd.read_csv(TWITTER_DAILY)
    else:
        raise IndexError("PLEASE check the flag: Google OR Twitter.")
    df = df.drop(['date'], axis=1)
    symptom_list = list(df.columns)
    return symptom_list

def load_data(config):
    if config.proxy == 'Google':
        folder = GOOGLE_FORECASTING_DATA
    elif config.proxy == 'Combined':
        folder = COMBINED_FORECASTING_DATA
    
    train_feature = pd.read_csv(f'{folder}/RF_data/RF_{config.proxy}_{config.gold_standard}_train_feature.csv', index_col=[0])
    test_feature = pd.read_csv(f'{folder}/RF_data/RF_{config.proxy}_{config.gold_standard}_test_feature.csv', index_col=[0])

    train_label = pd.read_csv(f'{folder}/RF_data/RF_{config.proxy}_{config.gold_standard}_train_label.csv', index_col=[0])
    test_label = pd.read_csv(f'{folder}/RF_data/RF_{config.proxy}_{config.gold_standard}_test_label.csv', index_col=[0])
    
    return train_feature, train_label, test_feature, test_label


def train(config, train_sliding_feature, train_sliding_label):
    """training model

    Args:
        config:
        train_sliding_feature (pd.DataFrame): input feature space
        train_sliding_label (pd.DataFrame): input label
    """
    rf = RandomForestClassifier(n_estimators=config.n_estimators, 
                                max_depth=config.max_depth, 
                                max_features=config.max_features, 
                                min_samples_leaf=config.min_samples_leaf, 
                                min_samples_split=config.min_samples_split)
    rf.fit(train_sliding_feature, train_sliding_label)
    return rf

def RF_train(config):
    """ Training procedure in Random Forest.

    Args:
        config (_type_): configuration of parameters.
    """
    x_train, y_train, x_test, y_test = load_data(config)
    if config.mode == "train":
        train_RF_model = train(config, x_train, y_train)
        save(config, train_RF_model, 'train')
        print(f'{config.proxy}_{config.gold_standard}_retrained model has been saved.')
    elif config.mode == "test":
        if config.proxy == 'Google':
            folder_output = GOOGLE_FORECASTING_REPORT_RF
        elif config.proxy == 'Combined':
            folder_output = COMBINED_FORECASTING_REPORT_RF

        saved_RF_model = load(config, 'train')
        pred_test = saved_RF_model.predict(x_test)
        prob_test = saved_RF_model.predict_proba(x_test)
        prob_test =  np.max(prob_test, axis=1)
        report = classification_report(y_test, pred_test, output_dict=True)

        print(report)
        print("Reduced feature model: Test finished.")

        # save prediction labels to csv file
        y_test['prediction'] = pred_test
        y_test['probability'] = prob_test
        y_test.to_csv(f'{folder_output}/{config.proxy}_{config.gold_standard}_test_prediction_trends.csv')
        
        # save test metrics to csv file
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv(f'{folder_output}/{config.proxy}_{config.gold_standard}_test_metrics.csv')
        
def save(config, model, flag):
    joblib.dump(model, f"{config.save_path}/{config.proxy}_{config.gold_standard}_{flag}.joblib")

def load(config, flag):
    load_model = joblib.load(f"{config.save_path}/{config.proxy}_{config.gold_standard}_{flag}.joblib")
    return load_model


if __name__ == '__main__':

    config = base_parser()
    if config.GPU != '-1':
        config.GPU_print = [int(config.GPU.split(',')[0])]
        os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU
        config.GPU = [int(i) for i in range(len(config.GPU.split(',')))]
    else:
        config.GPU = False

    np.random.seed(config.seed)
    feature_space = config.mode.split('_')[0]
    config.save_path = os.path.join(config.save_path, config.proxy, config.gold_standard, f'train_RF_models_forecast_{config.forecasting_horizon}days')

    os.makedirs(config.save_path, exist_ok=True)
    RF_train(config)

