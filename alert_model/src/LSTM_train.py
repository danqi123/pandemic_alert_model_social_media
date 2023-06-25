""" Module used to perform LSTM training process."""

import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
import shap
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as nnf
from torch.autograd import Variable
from pathlib import Path
from startup import GOOGLE_FORECASTING_REPORT_LSTM, COMBINED_FORECASTING_REPORT_LSTM
from LSTM_data_preprocessing import data_preprocessing_pipeline, create_dataset, create_label, read_label
from sklearn.preprocessing import MinMaxScaler

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
                        default= f'../models/LSTM/')
    parser.add_argument('--forecasting_horizon', type=int,
                        default=14)
    parser.add_argument('--training_length', type=int, 
                        default=28)
    parser.add_argument('--exp_name', type=str,
                        default='debug')
    parser.add_argument('--type', type=str,
                        default='Google_confirmed_cases',
                        choices=['Google_confirmed_cases',
                                'Google_hospitalization',
                                'Combined_hospitalization',
                                'Combined_confirmed_cases'])
    parser.add_argument('--mode', type=str,
                        default='test', choices=['train', 'test'])
    parser.add_argument('--from_best', type=bool,
                        default=True)
    parser.add_argument('--perform_shap', type=bool,
                        default=False)
    parser.add_argument('--testset_split', type=str,
                        default="2022-04-27")
    parser.add_argument('--lr', type=float, 
                        default=0.00118268283736384)
    parser.add_argument('--n_features', type=int,
                        default=20)
    parser.add_argument('--n_classes', type=int,
                        default=3)
    parser.add_argument('--n_hidden', type=int,
                        default=65)
    parser.add_argument('--n_layers', type=int,
                        default=3)
    parser.add_argument('--dropout', type=float,
                        default=0.1)
    parser.add_argument('--batch_size', type=int,
                        default=16)
    parser.add_argument('--num_epochs', type=int,
                        default=700)
    parser.add_argument('--epoch_init', type=int,
                        default=1)
    parser.add_argument('--save_freq', type=int, default=5)

    config = parser.parse_args("")
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

def load_dataset(config):
    splits = config.type.split('_')
    prox = splits[0]

    if len(splits) == 3: trends_ = splits[1] + '_' + splits[2]
    else: trends_ = splits[1]

    if prox == 'Google' or 'Combined':
        train, _, test = data_preprocessing_pipeline(proxy=prox, trend=trends_, training_length=config.training_length, forecasting_horizon=config.forecasting_horizon, testset_split=config.testset_split)

    else: raise IndexError('Please check proxy name: Google/Combined.')

    label = read_label(trend=trends_)

    config.n_features = train.shape[-1]
    config.feature_names = list(train.columns)

    # normalize data
    scaler = MinMaxScaler().fit(train)
    train_data = scaler.transform(train)
    train_data = pd.DataFrame(train_data)

    # create sliding window for feature values
    feature_window_train_ = create_dataset(train_data, config.training_length)

    # create sliding window for labels
    train_label_window = label[:len(train)]
    train_labels = create_label(train_label_window, config.training_length, config.forecasting_horizon)[:len(feature_window_train_)]
    if len(feature_window_train_) > len(train_labels):
        feature_window_train_ = feature_window_train_[:len(train_labels)]

    config.train_shap = [f.apply(pd.to_numeric) for f in feature_window_train_]
    train_set = list(zip(feature_window_train_, train_labels))
    
    dataset_train = Dataset_(train_set)
    # fit scaler to testset
    test_data = scaler.transform(test)
    test_data = pd.DataFrame(test_data)
    
    # create sliding window for feature train values
    feature_window_train_ = create_dataset(train_data, config.training_length)

    # create sliding window for train labels
    train_label_window = label[:len(train)]
    train_labels = create_label(train_label_window, config.training_length, config.forecasting_horizon)
    if len(feature_window_train_) > len(train_labels):
        feature_window_train_ = feature_window_train_[:len(train_labels)]

    config.train_shap = [f.apply(pd.to_numeric) for f in feature_window_train_]
    config.shap_train_plot_feature = []
    for t in config.train_shap:
        t = t.median(axis=0)
        t = list(t)
        t_df = pd.DataFrame(t).T
        config.shap_train_plot_feature.append(t)
    config.shap_train_plot_feature = np.array(config.shap_train_plot_feature)

    # create sliding window for feature test values
    feature_window_test_ = create_dataset(test_data, config.training_length)
    
    # create sliding window for test labels
    test_label_window = label[len(train)-1:]
    test_labels = create_label(test_label_window, config.training_length, config.forecasting_horizon)
    if len(feature_window_test_) > len(test_labels):
        feature_window_test_ = feature_window_test_[:len(test_labels)]

    config.test_shap = [f.apply(pd.to_numeric) for f in feature_window_test_]
    config.shap_test_plot_feature = []
    for t in config.test_shap:
        t = t.median(axis=0)
        t = list(t)
        t_df = pd.DataFrame(t).T
        config.shap_test_plot_feature.append(t)
    config.shap_test_plot_feature = np.array(config.shap_test_plot_feature)

    test_set = list(zip(feature_window_test_, test_labels))
    
    dataset_test = Dataset_(test_set)

    dataloader = get_loader(config, dataset_train)
    dataloader_test = get_loader(config, dataset_test, 1)

    # get test index for report data
    test_index = list(test.index)[config.training_length-1:][:len(test_labels)]

    return config, dataloader, dataloader_test, test_index

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

def LSTM_train(config):

    config, dataloader, dataloader_test, test_index = load_dataset(config)
    Train(config, dataloader, dataloader_test, test_index)


class Model(nn.Module):
    def __init__(self, config):
        super (Model, self).__init__()

        self.n_features = config.n_features
        self.n_classes = config.n_classes
        self.n_hidden = config.n_hidden
        self.n_layers = config.n_layers
        self.n_dropout = config.dropout

        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size = self.n_hidden,
            batch_first = True,
            num_layers = self.n_layers,
            dropout = self.n_dropout)

        self.classifier = nn.Linear(self.n_hidden, self.n_classes)

    def forward(self, x):

        _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden[-1])


class Train(object):
    def __init__(self, config, dataloader, dataloader_test, test_index):

        self.config = config
        self.device = torch.device('cuda:{}'.format(config.GPU[0])) if config.GPU else torch.device('cpu')
        self.dataloader = dataloader
        self.dataloader_test = dataloader_test
        self.test_index = test_index

        self.build_model()

        if self.config.mode == 'test':
            self.test()
            print('Test has finished')
        else:
            self.run()
            print('Training has finished')

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

    def test(self):

        epoch = self.config.epoch_init
        desc_bar = '[Test] Epoch: %d' % (epoch)

        progress_bar_test = tqdm(enumerate(self.dataloader_test),
                                unit_scale=True,
                                total=len(self.dataloader_test),
                                desc=desc_bar)

        predictions = []
        probability = []
        labels = []
        # batch size 1
        shap_dict = {}
        n = len(self.dataloader_test)
        for iter, data in progress_bar_test:

            Features = data[0].to(self.device)
            if not shap_dict:
                shap_dict = dict.fromkeys(config.feature_names, 0)
            label = data[1].to(self.device)

            outputs = self.Model(Features)
            prob = nnf.softmax(outputs, dim=1)
            top_p, _ = prob.topk(1, dim = 1)
            
            probability.append(torch.mode(top_p, 0)[0].item())
            prediction = torch.argmax(outputs, dim=1)
            predictions.append(torch.mode(prediction, 0)[0].item())
            labels.append(label.item())

        print('Labels:', labels)
        print('Predictions:',predictions)

        report_df = pd.DataFrame(columns=['label', 'prediction', 'probability'])
        report_df['label'] = labels
        report_df['prediction'] = predictions
        report_df['probability'] = probability

        report_df.index= self.test_index

        report_dict = {}
        report = classification_report(labels, predictions, output_dict=True)
        print(report)

        df_report = pd.DataFrame(report).transpose()
        splits = config.type.split('_')
        prox = splits[0]
        if prox == "Google":
            file = GOOGLE_FORECASTING_REPORT_LSTM  
        elif prox == "Twitter":
            file = TWITTER_FORECASTING_REPORT_LSTM
        elif prox == "Combined":
            file = COMBINED_FORECASTING_REPORT_LSTM
        df_report.to_csv(f'{file}/LSTM_{config.type}_metrics.csv')
        report_df.to_csv(f'{file}/LSTM_{config.type}_predictions.csv')

        if config.perform_shap:
            torch.set_grad_enabled(True)
            e = shap.DeepExplainer(self.Model, Variable(torch.from_numpy(np.array(config.train_shap, dtype="float32"))).to(self.device))
            torch.backends.cudnn.enabled=False
            shap_values = e.shap_values(Variable(torch.from_numpy(np.array(config.test_shap, dtype="float32"))).to(self.device))
            
            # select up-trends class
            shap_value_array = shap_values[1]

            sum_shap_mean = []
            for s in shap_value_array:
                s_ = np.mean(s, axis=0)
                sum_shap_mean.append(s_)
            final_shap_without_mean = np.array(sum_shap_mean)
            
            vals = np.abs(pd.DataFrame(final_shap_without_mean).values).mean(0)
            shap_importance = pd.DataFrame(list(zip(config.feature_names, vals)), columns=['col_name','feature_importance'])
            shap_importance.sort_values(by=['feature_importance'], ascending=False, inplace=True)
            shap_importance = shap_importance.set_index('col_name')
            print(shap_importance)

            layout=go.Layout(title=go.layout.Title(text="SHAP feature importance (uptrends)"))
            layout = go.Layout(title="SHAP Feature Importance", paper_bgcolor='white', plot_bgcolor='white')
            perm_columns = list(shap_importance.index)[:10][::-1]
            y_importance = list(shap_importance['feature_importance'])[:10][::-1]
            fig_1 = go.Figure(data=[go.Bar(x=y_importance, y=perm_columns, orientation='h')])#, orientation='h'
            fig_1.update_xaxes(color ='black', linewidth=1)
            fig_1.update_yaxes(color ='black', linewidth=1)
            fig_1.update_layout(xaxis_title='mean|SHAP value|', font=dict(size=20,color="RebeccaPurple"))
            fig_1.write_image(os.path.join(config.save_path_figures, f'bar_SHAP_feature_importance_uptrend_mean_test.jpg'), height=650, width=1200, scale=5)


    def run(self):
        global_steps = 0
        loss_train_history = []
        accuracy_train_history = []
        x_epoch = []
        
        # Epoch_init begins in 1
        for epoch in range(self.config.epoch_init, self.config.num_epochs + 1):
            
            desc_bar = '[Iter: %d] Epoch: %d/%d' % (
                global_steps, epoch, self.config.num_epochs)

            progress_bar = tqdm(enumerate(self.dataloader),
                                unit_scale=True,
                                total=len(self.dataloader),
                                desc=desc_bar)
            
            loss_func = nn.CrossEntropyLoss()

            loss_train = 0
            accuracy_train = 0
            num = 0

            # Training along dataset
            for iter, data in progress_bar:
                #self.Model.train()
                global_steps += 1

                Features = data[0].to(self.device)
                Labels = data[1].to(self.device).squeeze(-1)
                
                outputs = self.Model(Features)
                loss = loss_func(outputs, Labels)
                
                predictions = torch.argmax(outputs, dim=1)
                accuracy = predictions.eq(Labels.view_as(predictions)).sum().item()                

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_train += loss.item()
                accuracy_train += accuracy
                
            # import ipdb; ipdb.set_trace()

            loss_train /= len(self.dataloader.dataset)
            accuracy_train /= len(self.dataloader.dataset)
            loss_train_history.append(loss_train)
            accuracy_train_history.append(accuracy_train)
            x_epoch.append(epoch)

            if (epoch) % self.config.save_freq == 0:
                self.save(epoch, loss_train)
            if loss_train < self.best_loss:
                self.save(epoch, loss_train, best=True)
                self.best_loss = loss_train
                self.best_epoch = epoch

            self.save(epoch, loss_train)
               
        # plot the traning process
        fig = plt.figure()
        ax0 = fig.add_subplot(121, title="loss")
        ax1 = fig.add_subplot(122, title="accuracy")
        ax0.plot(x_epoch, loss_train_history, 'bo-', label='train_loss')
        ax1.plot(x_epoch, accuracy_train_history, 'ro-', label='train_accuracy')
        fig.savefig(os.path.join(config.save_path_figures, f'loss_accuracy.jpg'))

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

    config.save_path = os.path.join(config.save_path, config.type, f'LSTM_trained_models_forecast_{config.forecasting_horizon}days')

    config.save_path_samples = os.path.join(config.save_path, 'samples')
    config.save_path_models = os.path.join(config.save_path, 'models')
    config.save_path_losses = os.path.join(config.save_path, 'losses')
    config.save_path_figures = os.path.join(config.save_path, 'figures')

    os.makedirs(config.save_path, exist_ok=True)
    os.makedirs(config.save_path_samples, exist_ok=True)
    os.makedirs(config.save_path_models, exist_ok=True)
    os.makedirs(config.save_path_losses, exist_ok=True)
    os.makedirs(config.save_path_figures, exist_ok=True)

    LSTM_train(config)
