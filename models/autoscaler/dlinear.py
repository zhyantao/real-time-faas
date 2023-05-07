import math
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

device = torch.device('cpu')


class Parameters:
    def __init__(self):
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
        #
        # Number of epochs
        self.epochs = 10
        # Batch size
        self.batch_size = 256
        # Number of workers in DataLoader
        self.num_workers = 0
        # Define verbose
        self.verbose = True
        # Learning rate
        self.learning_rate = 1e-4
        # Trained model path
        self.model_path = 'models/DLinear.pth'

        # Data handling
        #
        # Filename
        self.filename = '../../dataset/electricity.csv'
        # Target series name
        self.targetSeries = 'OT'
        # Training-set percentage
        self.TrainingSetPercentage = 0.2
        # Data Log-transformation
        self.Transformation = True
        # Scaling {'Standard', 'MinMax', 'Robust'}
        self.Scaling = 'Standard'


args = Parameters()

start = time.time()
# Load data
df = pd.read_csv(args.filename)
print('[INFO] Data imported')
print('[INFO] Time: %.2f seconds' % (time.time() - start))
df.head(3)

# Convert Date to 'datetime64'
df['Date'] = df['Date'].astype('datetime64')
# Set index
df.set_index('Date', inplace=True)
# Keep only selected time-series
df = pd.DataFrame(df[[args.targetSeries]])
df.head(3)

# Split Training / Testing
idx = int(df.shape[0] * args.TrainingSetPercentage)
df_train = df[:idx].dropna()
df_test = df[idx:].dropna()

# Visualization
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 3))

df_train[args.targetSeries].plot(ax=ax, color='tab:blue')
df_test[args.targetSeries].plot(ax=ax, color='tab:orange')

plt.legend(['Training', 'Testing'], frameon=False, fontsize=14)
plt.ylabel(args.targetSeries, size=14)
plt.xlabel('Date', size=14)
plt.xticks(size=12)
plt.yticks(size=12)
plt.show()

# Fixing lag
df_test = pd.concat([df_train.iloc[-args.Lag:], df_test])

# Data Transformation
if args.Transformation == True:

    print('[INFO] Data transformation applied')

    VALUE = np.ceil(max(abs(-df.min().min()), 1.0))

    df_train = np.log(df_train + VALUE)
    df_test = np.log(df_test + VALUE)

else:
    print('[INFO] No data transformation applied.')

if args.Scaling == 'MinMax':
    print('[INFO] Scaling: MinMax')

    for feature in df.columns:
        if feature == args.targetSeries:
            continue
        print('Feature: ', feature)
        # Set scaler
        #
        scaler = MinMaxScaler()

        df_train[feature] = scaler.fit_transform(df_train[feature].to_numpy().reshape(-1, 1))
        df_test[feature] = scaler.transform(df_test[feature].to_numpy().reshape(-1, 1))

    # Scaling of Target Series
    #
    scaler = MinMaxScaler()
    df_train[args.targetSeries] = scaler.fit_transform(df_train[args.targetSeries].to_numpy().reshape(-1, 1))
    df_test[args.targetSeries] = scaler.transform(df_test[args.targetSeries].to_numpy().reshape(-1, 1))

elif args.Scaling == 'Robust':
    print('[INFO] Scaling: Robust')

    for feature in df.columns:
        if feature == args.targetSeries:
            continue
        print('Feature: ', feature)
        # Set scaler
        #
        scaler = RobustScaler()

        df_train[feature] = scaler.fit_transform(df_train[feature].to_numpy().reshape(-1, 1))
        df_test[feature] = scaler.transform(df_test[feature].to_numpy().reshape(-1, 1))

    # Scaling of Target Series
    #
    scaler = RobustScaler()
    df_train[args.targetSeries] = scaler.fit_transform(df_train[args.targetSeries].to_numpy().reshape(-1, 1))
    df_test[args.targetSeries] = scaler.transform(df_test[args.targetSeries].to_numpy().reshape(-1, 1))

elif args.Scaling == 'Standard':
    print('[INFO] Scaling: Standard')

    for feature in df.columns:
        if feature == args.targetSeries:
            continue
        print('Feature: ', feature)
        # Set scaler
        #
        scaler = StandardScaler()

        df_train[feature] = scaler.fit_transform(df_train[feature].to_numpy().reshape(-1, 1))
        df_test[feature] = scaler.transform(df_test[feature].to_numpy().reshape(-1, 1))

    # Scaling of Target Series
    #
    scaler = StandardScaler()

    df_train[args.targetSeries] = scaler.fit_transform(df_train[args.targetSeries].to_numpy().reshape(-1, 1))
    df_test[args.targetSeries] = scaler.transform(df_test[args.targetSeries].to_numpy().reshape(-1, 1))
else:
    print('[WARNING] Unknown data scaling. Standar scaling was selected')

    for feature in df.columns:
        if feature == args.targetSeries:
            continue
        print('Feature: ', feature)
        # Set scaler
        #
        scaler = StandardScaler()

        df_train[feature] = scaler.fit_transform(df_train[feature].to_numpy().reshape(-1, 1))
        df_test[feature] = scaler.transform(df_test[feature].to_numpy().reshape(-1, 1))

    # Scaling of Target Series
    #
    scaler = StandardScaler()

    df_train[args.targetSeries] = scaler.fit_transform(df_train[args.targetSeries].to_numpy().reshape(-1, 1))
    df_test[args.targetSeries] = scaler.transform(df_test[args.targetSeries].to_numpy().reshape(-1, 1))


def create_dataset(df=None, Lag=1, Horizon=1, targetSeries=None, overlap=1):
    if targetSeries is None:
        targetSeries = df.columns[-1]

    dataX, dataY, dataDate = [], [], []

    for i in tqdm(range(0, df.shape[0] + 1 - Lag - Horizon, overlap)):
        dataX.append(df.to_numpy()[i:(i + Lag)])
        dataY.append(df[targetSeries].to_numpy()[i + Lag: i + Lag + Horizon])
        dataDate.append(df.index[i + Lag: i + Lag + Horizon].tolist())

    return np.array(dataX), np.array(dataY), np.array(dataDate)


trainX, trainY, _ = create_dataset(df=df_train,
                                   Lag=args.Lag,
                                   Horizon=args.Horizon,
                                   targetSeries=args.targetSeries,
                                   overlap=1, )

testX, testY, testDate = create_dataset(df=df_test,
                                        Lag=args.Lag,
                                        Horizon=args.Horizon,
                                        targetSeries=args.targetSeries,
                                        overlap=1, )

# Last 10% of the training data will be used for validation
idx = int(0.9 * trainX.shape[0])
validX, validY = trainX[idx:], trainY[idx:]
trainX, trainY = trainX[:idx], trainY[:idx]
print('Training data shape:   ', trainX.shape, trainY.shape)
print('Validation data shape: ', validX.shape, validY.shape)
print('Testing data shape:    ', testX.shape, testY.shape)

# Reshaping
trainY = np.expand_dims(trainY, axis=-1)
validY = np.expand_dims(validY, axis=-1)
testY = np.expand_dims(testY, axis=-1)


class Data(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# Create training and test dataloaders
#
train_ds = Data(trainX, trainY)
valid_ds = Data(validX, validY)
test_ds = Data(testX, testY)

# Prepare Data-Loaders
#
train_dl = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers)
valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, num_workers=args.num_workers)
test_dl = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers)
#
print('[INFO] Data loaders were created')


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


# Initialize Neural Network
model = Model(args)
model.to(device)
print(model)

# Specify loss function
#
criterion = nn.MSELoss()

# Specify loss function
#
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=args.learning_rate)


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


# Early stopping
early_stopping = EarlyStopping(patience=100, min_delta=1e-5)

# LR scheduler
scheduler = LRScheduler(optimizer=optimizer,
                        patience=50,
                        min_lr=1e-10,
                        factor=0.5,
                        verbose=args.verbose)

# Store training and validation loss
Loss = {'Train': [],
        'Valid': []
        }

# Set number at how many iteration the training process (results) will be provided
#
batch_show = (train_dl.dataset.__len__() // args.batch_size // 5)

# Main loop - Training process
#
for epoch in range(1, args.epochs + 1):

    # Start timer
    start = time.time()

    # Monitor training loss
    #
    train_loss = 0.0
    valid_loss = 0.0

    ###################
    # Train the model #
    ###################
    batch_idx = 0
    for data, target in train_dl:

        # Clear the gradients of all optimized variables
        #
        optimizer.zero_grad()

        # Forward pass: compute predicted outputs by passing inputs to the model
        #
        if (device.type == 'cpu'):
            data = torch.tensor(data, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)
        else:
            data = torch.tensor(data, dtype=torch.float32).cuda()
            target = torch.tensor(target, dtype=torch.float32).cuda()

        outputs = model(data)

        # Calculate the loss
        #
        loss = criterion(outputs, target)

        # Backward pass: compute gradient of the loss with respect to model parameters
        #
        loss.backward()

        # Perform a single optimization step (parameter update)
        #
        optimizer.step()

        # Update running training loss
        #
        train_loss += loss.item() * data.size(0)

        # Increase batch_idx
        #
        batch_idx += 1

        # Info
        #
        if (args.verbose == True and batch_idx % batch_show == 0):
            print('> Epoch: {} [{:5.0f}/{} ({:.0f}%)]'.format(epoch, batch_idx * len(data), len(train_dl.dataset),
                                                              100. * batch_idx / len(train_dl)))

    # Print avg training statistics
    train_loss = train_loss / train_dl.dataset.X.shape[0]

    with torch.no_grad():
        for data, target in valid_dl:

            # Forward pass: compute predicted outputs by passing inputs to the model
            #
            if (device.type == 'cpu'):
                data = torch.tensor(data, dtype=torch.float32)
                target = torch.tensor(target, dtype=torch.float32)
            else:
                data = torch.tensor(data, dtype=torch.float32).cuda()
                target = torch.tensor(target, dtype=torch.float32).cuda()

            outputs = model(data)

            # Calculate the loss
            #
            loss = criterion(outputs, target)

            # update running training loss
            valid_loss += loss.item() * data.size(0)

    # Print avg training statistics
    #
    valid_loss = valid_loss / test_dl.dataset.X.shape[0]

    # Stop timer
    #
    stop = time.time()

    # Show training results
    #
    print('\n[INFO] Train Loss: {:.6f}\tValid Loss: {:.6f} \tTime: {:.2f}secs'.format(train_loss, valid_loss,
                                                                                      stop - start), end=' ')

    # Update best model
    #
    if epoch == 1:
        Best_score = valid_loss

        torch.save(model.state_dict(), args.model_path)
        print('(Model saved)\n')
    else:
        if Best_score > valid_loss:
            Best_score = valid_loss

            torch.save(model.state_dict(), args.model_path)
            print('(Model saved)\n')
        else:
            print('\n')

    # Store train/val loss
    #
    Loss['Train'] += [train_loss]
    Loss['Valid'] += [valid_loss]

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Learning rate scheduler
    #
    scheduler(valid_loss)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Early Stopping
    #
    if early_stopping(valid_loss):
        break

# Load best model
#
model.load_state_dict(torch.load(args.model_path))
model.eval()

print('[INFO] Model loaded')

pred = None
with torch.no_grad():
    for data, target in tqdm(test_dl):

        data = torch.tensor(data, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        if (pred is None):
            pred = model(data).numpy()
        else:
            pred = np.concatenate([pred, model(data).numpy()])

# Reshaping...
#
testY = testY.squeeze(-1)
pred = pred.squeeze(-1)

# Apply inverse scaling
#
for i in range(args.Horizon):
    testY[:, i] = scaler.inverse_transform(testY[:, i].reshape(-1, 1)).squeeze(-1)
    pred[:, i] = scaler.inverse_transform(pred[:, i].reshape(-1, 1)).squeeze(-1)

# Apply inverse transformation
#
if (args.Transformation == True):
    testY = np.exp(testY) - VALUE
    pred = np.exp(pred) - VALUE

print('[INFO] Feature: ', args.targetSeries)
print('------------------------------------------------')
Performance_Foresting_Model = {'RMSE': [], 'MAE': [], 'SMAPE': [], 'R2': []}


def smape(A, F):
    try:
        return (100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F))))
    except:
        return (np.NaN)


def rmse(A, F):
    try:
        return math.sqrt(metrics.mean_squared_error(A, F))
    except:
        return (np.NaN)


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

    return (MAE, RMSE, MAPE, SMAPE, R2)


for i in range(args.Horizon):
    Prices = pd.DataFrame([])

    Prices[args.targetSeries] = testY[:, i]
    Prices['Prediction'] = pred[:, i]

    # Evaluation
    #
    MAE, RMSE, MAPE, SMAPE, R2 = RegressionEvaluation(Prices)

    # Store results
    #
    Performance_Foresting_Model['RMSE'] += [RMSE]
    Performance_Foresting_Model['MAE'] += [MAE]
    Performance_Foresting_Model['SMAPE'] += [SMAPE]
    Performance_Foresting_Model['R2'] += [R2]

    # Present results
    #
    print('Horizon: %2i MAE %5.2f RMSE %5.2f SMAPE: %5.2f R2: %.2f' % (i + 1, MAE, RMSE, SMAPE, R2))

for i in range(args.Horizon):
    # Get actual values and predicted
    #
    Prices = pd.DataFrame([])

    Prices[args.targetSeries] = testY[:, i]
    Prices['Prediction'] = pred[:, i]

    # Calculate the residuals
    #
    res = (Prices[args.targetSeries] - Prices['Prediction']).to_numpy()

    # === Visualization ===
    #
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 2))

    # Plot residual histogram
    #
    ax[0].hist(res, bins=50)

    # Plot AutoCorrelation plot
    #
    plot_acf(res, ax=ax[1])
    ax[1].set_ylim([-1.05, 1.05])

subplots = [331, 332, 333, 334, 335, 336, 337, 338, 339]
plt.figure(figsize=(20, 15))

# Select random cases
RandomInstances = [random.randint(1, testY.shape[0]) for i in range(0, 9)]

for plot_id, i in enumerate(RandomInstances):
    plt.subplot(subplots[plot_id])
    plt.grid()
    #     plot_scatter(range(0, Lag), testX[i,:,0], color='b')
    plt.plot(testDate[i], testY[i], color='g', marker='o', linewidth=2)
    plt.plot(testDate[i], pred[i], color='r', marker='o', linewidth=2)

    plt.legend(['Actual values', 'Prediction'], frameon=False, fontsize=12)
    plt.ylim([np.min(testY[i]) - 10, np.max(testY[i]) + 10])
    plt.xticks(rotation=45)
plt.show()
