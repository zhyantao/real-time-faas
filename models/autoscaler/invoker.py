import os
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.autoscaler.bht_arima import BHTARIMA
from models.autoscaler.dlinear import Parameters, create_dataset, Data, Model, device, EarlyStopping, LRScheduler, \
    RegressionEvaluation
from models.autoscaler.lstm_v2 import LstmParam, LstmNetwork, ToyLossLayer
from models.utils.tools import get_one_machine


def run_lstm_v2(X, y):
    """加了差分模块的 LSTM 算法"""

    # 正则化数据
    ss = StandardScaler()
    std_X = ss.fit_transform(X)
    std_y = ss.fit_transform(y)

    # (1) 初始化 LSTM 模型
    lstm_param = LstmParam(mem_cell_ct=128, x_dim=X.shape[1])
    lstm_net = LstmNetwork(lstm_param)

    # 打印一些有用调试信息
    # print('X.shape = ', X.shape)
    # print('y.shape = ', y.shape)

    # (2) 训练 LSTM 模型
    for epoch in range(5000):
        # print("iter", "%2s" % str(epoch), end=": ")
        for i in range(len(std_y)):
            lstm_net.x_list_add(std_X[i])

        # (3) 预测和计算损失
        # print("y_pred = [" +
        #       ", ".join(["% 2.5f"
        #                  % ss.inverse_transform(lstm_net.lstm_node_list[i].state.h[0].reshape(1, -1))
        #                  for i in range(len(std_y))]) +
        #       "]", end=", ")
        loss = lstm_net.y_list_is(std_y, ToyLossLayer)  # 计算损失
        # print("loss:", "%.3e" % loss)

        # (4) 更新模型
        lstm_param.apply_diff(lr=0.01)
        lstm_net.x_list_clear()  # 清理掉原来的参数

    # (5) 数据后处理：还原 y
    # origin_y = ss.inverse_transform(std_y)
    y_hat = (np.zeros_like(std_y)).reshape(-1)
    for i in range(len(std_y)):
        pred = ss.inverse_transform(lstm_net.lstm_node_list[i].state.h[0].reshape(1, -1))
        y_hat[i] = pred[0]

    return y_hat


def run_bht_arima(X, y):
    """调用 BHT ARIMA 算法（单步预测）"""

    # 正则化数据
    # ss = MinMaxScaler()  # SVD 默认包含正则化：https://stackoverflow.com/a/46025739/16733647
    # std_X = ss.fit_transform(X)
    # std_y = ss.fit_transform(y)

    # parameters setting
    n_samples = X.shape[0]

    # print('y = ', y)
    p = 3  # p-order
    d = 2  # d-order
    q = 1  # q-order
    taus = [n_samples, 5]  # MDT-rank
    Rs = [5, 5]  # tucker decomposition ranks
    epochs = 5000  # iterations
    tol = 0.001  # stop criterion
    Us_mode = 4  # orthogonality mode

    # Run program
    # result's shape: (ITEM, TIME+1) ** only one step forecasting **
    model = BHTARIMA(X, p, d, q, taus, Rs, epochs, tol, verbose=0, Us_mode=Us_mode)
    pred, _ = model.run()
    y_hat = pred[..., -1]

    return y_hat


def call_dlinear(column_name, machine_idx):
    args = Parameters(column_name)

    start = time.time()
    # Load data
    df = pd.read_csv(args.filename)
    print('[INFO] Data imported')
    print('[INFO] Time: %.2f seconds' % (time.time() - start))

    # (1) 每次从文件中读取一个 task 的资源需求变化
    machine, next_idx = get_one_machine(df, machine_idx)
    print(machine)
    machine_name = machine['machine_id'].loc[machine.index[0]]
    print(machine_name)

    # # Convert Date to 'datetime64'
    # df['Date'] = df['Date'].astype('datetime64')
    # # Set index
    # df.set_index('Date', inplace=True)
    # Keep only selected time-series
    df = pd.DataFrame(machine[[args.targetSeries]])

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
        print('[WARNING] Unknown data scaling. Standard scaling was selected')

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
    batch_show = (train_dl.dataset.__len__() // args.batch_size)

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
            if device.type == 'cpu':
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

            # # Info
            # #
            # if args.verbose == True and batch_idx % batch_show == 0:
            #     print('> Epoch: {} [{:5.0f}/{} ({:.0f}%)]'.format(epoch, batch_idx * len(data), len(train_dl.dataset),
            #                                                       100. * batch_idx / len(train_dl)))

        # Print avg training statistics
        train_loss = train_loss / train_dl.dataset.X.shape[0]

        with torch.no_grad():
            for data, target in valid_dl:

                # Forward pass: compute predicted outputs by passing inputs to the model
                #
                if device.type == 'cpu':
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
        if not os.path.exists(os.path.dirname(args.model_path)):
            os.makedirs(os.path.dirname(args.model_path))

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
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    print('[INFO] Model loaded')

    pred = None
    with torch.no_grad():
        for data, target in tqdm(test_dl):

            data = torch.tensor(data, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)

            if pred is None:
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
    if args.Transformation == True:
        testY = np.exp(testY) - VALUE
        pred = np.exp(pred) - VALUE

    print('[INFO] Feature: ', args.targetSeries)
    print('------------------------------------------------')
    Performance_Foresting_Model = {'RMSE': [], 'MAE': [], 'SMAPE': [], 'R2': []}

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

    plt.figure()
    plt.plot(pred[:, 3], label='Pred.')
    plt.plot(testY[:, 3], label='True')
    plt.legend()
    plt.show()

    return pred[:, 3], testY[:, 3]


def run_dlinear(machine_idx):
    cpu_y_hat, cpu_y_test = call_dlinear('cpu_util_percent', machine_idx)
    mem_y_hat, mem_y_test = call_dlinear('mem_util_percent', machine_idx)
    y_hat = []
    y_test = []
    for i in range(0, len(cpu_y_hat)):
        y_hat.append([cpu_y_hat[i], mem_y_hat[i]])
        y_test.append([cpu_y_test[i], mem_y_test[i]])
    return np.array(y_hat), np.array(y_test)
