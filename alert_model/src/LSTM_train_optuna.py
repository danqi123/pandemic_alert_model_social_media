# Scripts used to perform LSTM_optuna_hyperparameter tuning.

import os
import warnings
import numpy as np
import pandas as pd
import optuna
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
from LSTM_data_preprocessing import data_preprocessing_pipeline, create_dataset, create_label, read_label
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)
home_dir = str(Path.home())


def base_parser():

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--GPU', type=str, default='0',
                        help='Set -1 for CPU running')
    parser.add_argument('--seed', type=int,
                        default=2)
    parser.add_argument('--save_path', type=str,
                        default=f'../models/LSTM/')
    parser.add_argument('--exp_name', type=str,
                        default='debug')
    parser.add_argument('--type', type=str,
                        default='Google_hospitalization',
                        choices=['Google_hospitalization',
                                'Google_confirmed_cases',
                                'Combined_confirmed_cases',
                                'Combined_hospitalization'])
    parser.add_argument('--forecasting_horizon', type=int,
                        default=14)
    parser.add_argument('--training_length', type=int,
                        default=28)
    parser.add_argument('--mode', type=str,
                        default='train', choices=['train', 'test'])
    parser.add_argument('--testset_split', type=str,
                        default="2022-04-27")
    parser.add_argument('--from_best', type=bool,
                        default=False)
    parser.add_argument('--lr', type=float,
                        default=0.0002286620653812753)
    parser.add_argument('--n_features', type=int,
                        default=5)
    parser.add_argument('--n_classes', type=int,
                        default=3)
    parser.add_argument('--n_hidden', type=int,
                        default=115)
    parser.add_argument('--n_layers', type=int,
                        default=2)
    parser.add_argument('--dropout', type=float,
                        default=0.1)
    parser.add_argument('--batch_size', type=int,
                        default=32)
    parser.add_argument('--num_epochs', type=int,
                        default=700)
    parser.add_argument('--cv_initial_window', type=int,
                        default=90)
    parser.add_argument('--cv_step_length', type=int,
                        default=70)
    parser.add_argument('--cv_test_window', type=int,
                        default=30)
    parser.add_argument('--epoch_init', type=int,
                        default=1)
    parser.add_argument('--save_freq', type=int,
                        default=5)

    config = parser.parse_args()
    return config

def get_loader(config, dataset, bs=None):

    pm = True if torch.cuda.is_available() else False

    if bs is None:
        bs = config.batch_size
        shuffle = False
    else:
        shuffle = False

    dataloader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = bs,
        shuffle = shuffle,
        pin_memory=pm,
        drop_last = False)
    return dataloader

def timeseries_cv(window: int, forecasting_horizon: list, steps: int):
    cv = SlidingWindowSplitter(window_length=window, fh=forecasting_horizon,
                               step_length=steps)
    return cv

class Dataset_(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        feats = []
        labels = []
        for i in range(len(self.sequences)):
            feat = torch.from_numpy(self.sequences[i][0].to_numpy()).float()
            label = torch.Tensor(self.sequences[i][1:]).long()

            feats.append(feat)
            labels.append(label)

        self.feats = torch.stack(feats, 0)
        self.labels = torch.stack(labels, 0)

    def __getitem__(self, idx):
        feat = self.feats[idx]
        label = self.labels[idx]

        return feat, label

    def __len__(self):
        return len(self.feats)

def main_cv(trial, config):

    splits = config.type.split('_')
    prox = splits[0]
    if len(splits) == 3:
        trends_ = splits[1] + '_' + splits[2]
    else:
        trends_ = splits[1]

    # load dataset
    if prox == 'Google' or 'Combined':
        train_val, train_index, _ = data_preprocessing_pipeline(proxy=prox, trend=trends_, training_length=config.training_length, forecasting_horizon=config.forecasting_horizon, testset_split=config.testset_split)
    else:
        raise IndexError('Please check proxy name: Google/Combined')

    config.n_features = train_val.shape[-1]

    # set time series CV 
    forecasting_horizon = list(range(config.training_length + config.forecasting_horizon-1, config.training_length + config.forecasting_horizon -1 + config.cv_test_window))
    fold = timeseries_cv(window = config.cv_initial_window, forecasting_horizon=forecasting_horizon, steps = config.cv_step_length)

    loss = []
    len_cv = 0
    label = read_label(trend=trends_)
    cv_dict = {}

    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(train_index), 1):
        subset_train = train_val.iloc[train_idx[0]:train_idx[-1]+config.training_length+1, :]
        # normalize data
        scaler = MinMaxScaler().fit(subset_train)
        train_data = scaler.transform(subset_train)
        train_data = pd.DataFrame(train_data, index = subset_train.index)

        feature_window_train_ = create_dataset(train_data, config.training_length)

        subset_label = label[train_idx[0]:train_idx[-1] + config.training_length + config.forecasting_horizon]
        subset_labels = create_label(subset_label, config.training_length, config.forecasting_horizon)[:len(feature_window_train_)]

        train_set = list(zip(feature_window_train_, subset_labels))
        
        train_loader = Dataset_(train_set)
        train_loader = get_loader(config, train_loader)
        subset_val = train_val.iloc[valid_idx[0]:valid_idx[-1]+config.training_length+1, :]

        # normalize validation data
        val_data = scaler.transform(subset_val)
        val_data = pd.DataFrame(val_data, index = subset_val.index)
        feature_window_val_ = create_dataset(val_data, config.training_length)

        subset_val_label = label[valid_idx[0]:valid_idx[-1]+config.training_length +config.forecasting_horizon]
        subset_val_labels = create_label(subset_val_label, config.training_length, config.forecasting_horizon)[:len(feature_window_val_)]

        val_set = list(zip(feature_window_val_, subset_val_labels))

        bs_val = len(valid_idx)
        valid_loader = Dataset_(val_set)
        valid_loader = get_loader(config, valid_loader, bs_val)

        config.batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        config.n_hidden = trial.suggest_int('n_hidden', 60, 120)
        config.n_layers = trial.suggest_int('n_layers', 2, 4)
        config.num_epochs = trial.suggest_categorical('num_epochs', [400, 500, 600, 700])
        config.lr = trial.suggest_loguniform('lr', 0.0004, 0.002)
        config.dropout = trial.suggest_float('dropout', 0, 0.2, step=0.1)

        train_ = Train(config, train_loader, valid_loader)

        best_loss, best_epoch = train_.run()
        trial.set_user_attr("nepochs", best_epoch)
        loss.append(best_loss)
        len_cv = fold_idx

    mean_loss = np.mean(loss)

    return mean_loss

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.n_features = config.n_features
        self.n_classes = config.n_classes
        self.n_hidden = config.n_hidden
        self.n_layers = config.n_layers
        self.n_dropout = config.dropout

        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size = self.n_hidden,
            batch_first=True,
            num_layers = self.n_layers,
            dropout = self.n_dropout)

        self.classifier = nn.Linear(self.n_hidden, self.n_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden[-1])

class Train(object):
    def __init__(self, config, dataloader, dataloader_val):

        self.config = config
        self.device = torch.device('cuda:{}'.format(config.GPU[0])) if config.GPU else torch.device('cpu')
        self.dataloader = dataloader
        self.dataloader_val = dataloader_val
        self.build_model()


    def build_model(self):

        self.Model = Model(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(self.Model.parameters(), self.config.lr)

        if self.config.epoch_init != 1 or self.config.from_best:
            self.load_models()
        else:
            self.best_loss = 10e15
            self.best_epoch = 1

    def load_models(self):

        if self.config.from_best:
            weights = torch.load(os.path.join(
                self.config.save_path_models, 'Best.pth'),
                map_location=self.device)
            self.config.epoch_init = weights['Epoch']
            epoch = self.config.epoch_init
        else:
            epoch = self.config.epoch_init
            weights = torch.load(os.path.join(
                self.config.save_path_models, 'Ckpt_%d.pth'%(epoch)),
                map_location=self.device)

        self.best_loss = weights['Loss']
        self.Model.load_state_dict(weights['Model'])

        if 'train' in self.config.mode:
            self.optimizer.load_state_dict(weights['Opt'])

        print('Models have loaded from epoch:', epoch)

    def save(self, epoch, loss, best=False):

        weights = {}
        weights['Model'] = self.Model.state_dict()
        weights['Opt'] = self.optimizer.state_dict()
        weights['Loss'] = loss
        if best:
            weights['Epoch'] = epoch
            torch.save(weights,
                os.path.join(self.config.save_path_models, 'Best.pth'))
        else:
            torch.save(weights,
                os.path.join(self.config.save_path_models, 'Ckpt_%d.pth'%(epoch)))

    def run(self):
        global_steps = 0
        loss_train_history = []
        loss_val_history = []
        accuracy_train_history = []
        accuracy_val_history = []
        x_epoch = []
        #trial_num = 0
        # Epoch_init begins in 1

        #tb = SummaryWriter()
        for epoch in range(self.config.epoch_init, self.config.num_epochs + 1):

            self.Model.train()

            desc_bar = '[Iter: %d] Epoch: %d/%d' % (
                global_steps, epoch, self.config.num_epochs)

            progress_bar = tqdm(enumerate(self.dataloader),
                                unit_scale=True,
                                total=len(self.dataloader),
                                desc=desc_bar)

            loss_func = nn.CrossEntropyLoss()
            epoch_loss_val = 0
            epoch_loss_train = 0
            correct_train = 0

            # Training along dataset
            for iter, data in progress_bar:
                #self.Model.train()
                global_steps += 1

                Features = data[0].to(self.device)
                Labels = data[1].to(self.device).squeeze(-1)

                outputs = self.Model(Features)
                loss = loss_func(outputs, Labels)
                predictions = torch.argmax(outputs, dim=1)
                correct_train += predictions.eq(Labels.view_as(predictions)).sum().item()


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss_train += loss.item()

                # ===============================================
                # =============== Validation ====================
                # ===============================================

            correct_val = 0
            desc_bar = '[VAL] Epoch: %d' % (epoch)
            progress_bar_val = tqdm(enumerate(self.dataloader_val),
                                        unit_scale=True,
                                        total=len(self.dataloader_val),
                                        desc=desc_bar)

            self.Model.eval()

            for iter, data in progress_bar_val:
                Features_val = data[0].to(self.device)
                Labels_val = data[1].to(self.device).squeeze(-1)

                outputs_val = self.Model(Features_val)
                loss_val = loss_func(outputs_val, Labels_val)
                epoch_loss_val += loss_val.item()
                predictions_val = torch.argmax(outputs_val, dim=1)
                correct_val += predictions_val.eq(Labels_val.view_as(predictions_val)).sum().item()

            epoch_loss_train /= len(self.dataloader.dataset)
            epoch_loss_val /= len(self.dataloader_val.dataset)

            epoch_accuracy_train = correct_train / len(self.dataloader.dataset)
            epoch_accuracy_val = correct_val / len(self.dataloader_val.dataset)

            loss_train_history.append(epoch_loss_train)
            loss_val_history.append(epoch_loss_val)
            accuracy_train_history.append(epoch_accuracy_train)
            accuracy_val_history.append(epoch_accuracy_val)
            x_epoch.append(epoch)

            if (epoch) % self.config.save_freq == 0:
               self.save(epoch, epoch_loss_val)


            if epoch_loss_val < self.best_loss:
               self.save(epoch, epoch_loss_val, best=True)
               self.best_loss = epoch_loss_val
               self.best_epoch = epoch

        # plot training and validation history
        fig = plt.figure()
        ax0 = fig.add_subplot(121, title="loss")
        ax1 = fig.add_subplot(122, title="accuracy")
        ax0.plot(x_epoch, loss_train_history, 'bo-', label='train_loss')
        ax0.plot(x_epoch, loss_val_history, 'ro-', label='validation_loss')
        ax1.plot(x_epoch, accuracy_train_history, 'bo-', label='train_accuracy')
        ax1.plot(x_epoch, accuracy_val_history, 'ro-', label='validation_accuracy')
        fig.savefig(os.path.join(config.save_path_figures, f'{loss_val_history[0]}_loss_accuracy.jpg'))

        return self.best_loss, self.best_epoch

def max_trial_callback(study, trial):
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE or t.state == optuna.trial.TrialState.RUNNING])
    if n_complete > 89:
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
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    config.save_path = os.path.join(config.save_path, config.type, f'LSTM_models_forecast_{config.forecasting_horizon}days')
    config.save_path_samples = os.path.join(config.save_path, 'samples')
    config.save_path_models = os.path.join(config.save_path, 'models')
    config.save_path_figures = os.path.join(config.save_path, 'figures')
    config.save_path_losses = os.path.join(config.save_path, 'losses')

    os.makedirs(config.save_path, exist_ok=True)
    os.makedirs(config.save_path_samples, exist_ok=True)
    os.makedirs(config.save_path_models, exist_ok=True)
    os.makedirs(config.save_path_losses, exist_ok=True)
    os.makedirs(config.save_path_figures, exist_ok=True)


    study = optuna.create_study(study_name=config.exp_name,
                                sampler=optuna.samplers.TPESampler(),
                                direction='minimize')

    study.optimize(lambda trial: main_cv(trial, config), n_jobs=1, callbacks=[max_trial_callback])

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    print('Avg loss', trial.value)

    name_csv = os.path.join(config.save_path, 'Best_hyperparameters.csv')

    print('  Params: ')
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    dic = dict(trial.params)
    df = pd.DataFrame.from_dict(data=dic, orient='index').to_csv(name_csv, header=False)
