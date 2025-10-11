import numpy as np
import pandas as pd

def log_string(log, string):
    """
    Write a string to the log file and print it to the console.
    """
    log.write(string + '\n')
    log.flush()
    print(string)

def metric(pred, label):
    """
    Calculate performance metrics: MAE, RMSE, MAPE.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
        
    return mae, rmse, mape

def seq2instance(data, P, Q):
    """
    Convert sequence data into instances for training/testing.
    :param data: sequence data.
    :param P: history steps.
    :param Q: prediction steps.
    :return: x, y
    """
    num_step, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape=(num_sample, P, dims))
    y = np.zeros(shape=(num_sample, Q, dims))
    for i in range(num_sample):
        x[i] = data[i: i + P]
        y[i] = data[i + P: i + P + Q]
    return x, y

def load_data(args):
    """
    Load and preprocess data.
    """
    # Traffic data
    df = pd.read_hdf(args.traffic_file)
    traffic = df.values

    # Split data
    num_step = df.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    
    train = traffic[:train_steps]
    val = traffic[train_steps: train_steps + val_steps]
    test = traffic[-test_steps:]

    # Create instances
    trainX, trainY = seq2instance(train, args.P, args.Q)
    valX, valY = seq2instance(val, args.P, args.Q)
    testX, testY = seq2instance(test, args.P, args.Q)

    # Normalization
    mean, std = np.mean(trainX), np.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std

    # Spatial Embedding
    with open(args.se_file, mode='r') as f:
        lines = f.readlines()
        temp = lines[0].split(' ')
        num_vertex, dims = int(temp[0]), int(temp[1])
        se = np.zeros((num_vertex, dims), dtype=np.float32)
        for line in lines[1:]:
            temp = line.split(' ')
            index = int(temp[0])
            se[index] = [float(i) for i in temp[1:]]

    # Temporal Embedding
    time_index = df.index
    dayofweek = np.reshape(time_index.weekday, newshape=(-1, 1))
    timeofday = (time_index.hour * 3600 + time_index.minute * 60 + time_index.second) // (
            args.time_slot * 60)
    timeofday = np.reshape(timeofday, newshape=(-1, 1))
    time = np.concatenate((dayofweek, timeofday), axis=-1)

    # Create temporal instances
    train_te = seq2instance(time[:train_steps], args.P, args.Q)
    train_te = np.concatenate(train_te, axis=1).astype(np.int32)
    val_te = seq2instance(time[train_steps: train_steps + val_steps], args.P, args.Q)
    val_te = np.concatenate(val_te, axis=1).astype(np.int32)
    test_te = seq2instance(time[-test_steps:], args.P, args.Q)
    test_te = np.concatenate(test_te, axis=1).astype(np.int32)

    return (trainX, train_te, trainY, valX, val_te, valY, testX, test_te, testY,
            se, mean, std)
