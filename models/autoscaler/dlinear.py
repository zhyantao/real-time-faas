import math
import warnings

import numpy as np
import torch
from sklearn import metrics
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from models.utils.parameters import args

warnings.filterwarnings("ignore")

device = torch.device('cpu')


class Parameters:
    def __init__(self, column_name):
        self.description = 'DLinear model for time-series forecasting'

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # Neural network model parameters
        #
        # Input sequence length - look-back
        self.Lag = 12
        # Prediction sequence length
        self.Horizon = 4
        #
        self.individual = False
        self.enc_in = 1
        self.kernel_size = 25

        # Training parameters
        # Number of epochs
        self.epochs = 5000
        # Batch size
        self.batch_size = 32
        # Number of workers in DataLoader
        self.num_workers = 0
        # Define verbose
        self.verbose = True
        # Learning rate
        self.learning_rate = 1e-4
        # Trained model path
        self.model_path = args.result_saving_path + 'weights/DLinear.pth'

        # Data handling
        # Filename
        self.filename = './dataset/selected_container_usage.csv'
        # Target series name
        self.targetSeries = column_name
        # Training-set percentage
        self.TrainingSetPercentage = 0.37
        # Data Log-transformation
        self.Transformation = True
        # Scaling {'Standard', 'MinMax', 'Robust'}
        self.Scaling = 'Standard'


def create_dataset(df=None, Lag=1, Horizon=1, targetSeries=None, overlap=1):
    if targetSeries is None:
        targetSeries = df.columns[-1]

    dataX, dataY, dataDate = [], [], []

    for i in tqdm(range(0, df.shape[0] + 1 - Lag - Horizon, overlap)):
        dataX.append(df.to_numpy()[i:(i + Lag)])
        dataY.append(df[targetSeries].to_numpy()[i + Lag: i + Lag + Horizon])
        dataDate.append(df.index[i + Lag: i + Lag + Horizon].tolist())

    return np.array(dataX), np.array(dataY), np.array(dataDate)


class Data(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# Model: DLinear
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    DLinear
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.Lag = configs.Lag
        self.Horizon = configs.Horizon

        # Decomposition Kernel Size
        kernel_size = configs.kernel_size
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            self.Linear_Decoder = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.Lag, self.Horizon))
                self.Linear_Seasonal[i].weight = nn.Parameter((1 / self.Lag) * torch.ones([self.Horizon, self.Lag]))
                self.Linear_Trend.append(nn.Linear(self.Lag, self.Horizon))
                self.Linear_Trend[i].weight = nn.Parameter((1 / self.Lag) * torch.ones([self.Horizon, self.Lag]))
                self.Linear_Decoder.append(nn.Linear(self.Lag, self.Horizon))
        else:
            self.Linear_Seasonal = nn.Linear(self.Lag, self.Horizon)
            self.Linear_Trend = nn.Linear(self.Lag, self.Horizon)
            self.Linear_Decoder = nn.Linear(self.Lag, self.Horizon)
            self.Linear_Seasonal.weight = nn.Parameter((1 / self.Lag) * torch.ones([self.Horizon, self.Lag]))
            self.Linear_Trend.weight = nn.Parameter((1 / self.Lag) * torch.ones([self.Horizon, self.Lag]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.Horizon],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.Horizon],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=3, min_delta=0.):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss

        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss

        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1

            if self.counter >= self.patience:
                print(f'[INFO] Early stopping')
                return (True)
            else:
                return (False)


class LRScheduler:
    """
    Learning rate scheduler. If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    given `factor`.
    """

    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5, verbose=True):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.verbose = verbose

        #         self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = self.optimizer,
        #                                                                        steps     = self.patience,
        #                                                                        verbose   = self.verbose )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=self.verbose
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


def smape(A, F):
    try:
        return (100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F))))
    except:
        return np.NaN


def rmse(A, F):
    try:
        return math.sqrt(metrics.mean_squared_error(A, F))
    except:
        return np.NaN


def RegressionEvaluation(Prices):
    '''
    Parameters
    ----------
    Y : TYPE
        Real prices.
    Pred : TYPE
        Predicted prices.
    Returns
    -------
    MAE : TYPE
        Mean Absolute Error.
    RMSE : TYPE
        Root Mean Square Error.
    MAPE : TYPE
        Mean Absolute Percentage Error.
    R2   : TYPE
        R2 correlation
    '''

    SeriesName = Prices.columns[0]
    Prediction = Prices.columns[1]

    Y = Prices[SeriesName].to_numpy()
    Pred = Prices[Prediction].to_numpy()

    MAE = metrics.mean_absolute_error(Y, Pred)
    RMSE = math.sqrt(metrics.mean_squared_error(Y, Pred))
    try:
        MAPE = np.mean(np.abs((Y - Pred) / Y)) * 100.0
    except:
        MAPE = np.NaN

    SMAPE = smape(Y, Pred)
    R2 = metrics.r2_score(Y, Pred)

    return MAE, RMSE, MAPE, SMAPE, R2
