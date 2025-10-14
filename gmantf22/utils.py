import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import mean_absolute_error, mean_squared_error


class GMANConfig(BaseModel):
    time_slot: int = Field(5, description="time interval")
    P: int = Field(12, description="history steps")
    Q: int = Field(12, description="prediction steps")
    L: int = Field(5, description="number of STAtt Blocks")
    K: int = Field(8, description="number of attention heads")
    d: int = Field(8, description="dims of each head attention outputs")
    train_ratio: float = Field(0.7, description="training set ratio")
    val_ratio: float = Field(0.1, description="validation set ratio")
    test_ratio: float = Field(0.2, description="testing set ratio")
    batch_size: int = Field(48, description="batch size")
    max_epoch: int = Field(100, description="epoch to run")
    patience: int = Field(10, description="patience for early stop")
    learning_rate: float = Field(0.001, description="initial learning rate")
    decay_epoch: int = Field(5, description="decay epoch")
    traffic_file: str = Field("./data/METR-LA/metr-la.h5", description="traffic file")
    SE_file: str = Field(
        "./data/METR-LA/SE(METR).txt", description="spatial embedding file"
    )
    model_file: str = Field(
        "./models/GMAN.weights.h5", description="save the model to disk"
    )
    log_file: str = Field("./log/log", description="log file")
    use_mixed_precision: bool = Field(True, description="use mixed precision training")


def log_string(log: logging.Logger, string: str):
    """
    Write a string to the log file and print it to the console.
    """
    log.info(string)
    print(string)


def metric(pred: np.ndarray, label: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate performance metrics: MAE, RMSE, MAPE.
    This function replicates the original's masking behavior for zero values in labels.
    """
    # Mask for non-zero labels, crucial for traffic data where 0 indicates no data.
    mask = label != 0

    # If all labels are zero, metrics are undefined and should be 0.
    if not np.any(mask):
        return 0.0, 0.0, 0.0

    # Filter out the zero-valued data points from both prediction and label
    pred_masked = pred[mask]
    label_masked = label[mask]

    # Calculate metrics using the masked data
    mae = mean_absolute_error(label_masked, pred_masked)
    rmse = np.sqrt(mean_squared_error(label_masked, pred_masked))

    # MAPE calculation, avoiding division by zero.
    mape = np.mean(np.abs(np.divide(label_masked - pred_masked, label_masked)))

    return float(mae), float(rmse), float(mape)


def seq2instance(data: np.ndarray, P: int, Q: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert sequence data into instances for training/testing.
    :param data: sequence data of shape (num_step, dims).
    :param P: history steps.
    :param Q: prediction steps.
    :return: A tuple of (x, y) where x is input instances and y is label instances.
    """
    num_step, dims = data.shape
    num_sample = num_step - P - Q + 1
    if num_sample < 1:
        raise ValueError(
            f"Not enough data to create samples. Required {P + Q} steps, but got {num_step}."
        )

    x = np.zeros((num_sample, P, dims))
    y = np.zeros((num_sample, Q, dims))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y


def load_data(
    args: GMANConfig,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    """
    Load and preprocess data with comprehensive normalization for mixed precision.
    All features are normalized to prevent overflow in mixed_float16.
    """
    traffic_path = Path(args.traffic_file)
    se_path = Path(args.SE_file)

    if not traffic_path.exists():
        raise FileNotFoundError(f"Traffic file not found: {traffic_path}")
    if not se_path.exists():
        raise FileNotFoundError(f"SE file not found: {se_path}")

    # Load traffic data
    df = pd.read_hdf(traffic_path)
    Traffic = df.values.astype(np.float32)

    # Ensure index is DatetimeIndex for time-based operations
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")
    time_index: pd.DatetimeIndex = df.index

    print(f"Loaded traffic data shape: {Traffic.shape}")
    print(f"Time range: {time_index.min()} to {time_index.max()}")

    # Split data
    num_step = df.shape[0]
    train_steps = round(args.train_ratio * num_step)
    val_steps = round(args.val_ratio * num_step)
    test_steps = num_step - train_steps - val_steps

    for name, steps in [
        ("Training", train_steps),
        ("Validation", val_steps),
        ("Test", test_steps),
    ]:
        if steps < args.P + args.Q:
            raise ValueError(
                f"{name} set too small: {steps} steps. Needs at least {args.P + args.Q}."
            )

    train = Traffic[:train_steps]
    val = Traffic[train_steps : train_steps + val_steps]
    test = Traffic[-test_steps:]

    # Create instances
    trainX, trainY = seq2instance(train, args.P, args.Q)
    valX, valY = seq2instance(val, args.P, args.Q)
    testX, testY = seq2instance(test, args.P, args.Q)

    # Normalize features using training set statistics
    mean, std = np.mean(trainX), np.std(trainX)
    std = np.maximum(std, 1e-5)  # Avoid division by zero

    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std

    # Load and normalize spatial embeddings
    try:
        # Use np.loadtxt for efficient parsing of the text file
        SE = np.loadtxt(se_path, skiprows=1)[:, 1:].astype(np.float32)

        # Normalize SE for stability
        se_mean, se_std = np.mean(SE), np.std(SE)
        se_std = np.maximum(se_std, 1e-5)
        SE = (SE - se_mean) / se_std

        print(f"Loaded spatial embedding: {SE.shape}")
        print(f"SE normalized - mean: {se_mean:.4f}, std: {se_std:.4f}")
    except Exception as e:
        raise ValueError(f"Error loading or processing SE file '{se_path}': {e}")

    # Create temporal embeddings from the DatetimeIndex
    day_of_week = np.reshape(time_index.weekday, (-1, 1))

    freq_seconds = args.time_slot * 60

    time_of_day = (
        time_index.hour * 3600 + time_index.minute * 60 + time_index.second
    ) // freq_seconds
    time_of_day = np.reshape(time_of_day, (-1, 1))

    TE = np.concatenate((day_of_week, time_of_day), axis=-1).astype(np.int32)
    print(
        f"Time embedding range - day: [0, 6], time_slot: [0, {int(time_of_day.max())}]"
    )

    # Create temporal instances
    train_te_x, train_te_y = seq2instance(TE[:train_steps], args.P, args.Q)
    train_te = np.concatenate((train_te_x, train_te_y), axis=1)

    val_te_x, val_te_y = seq2instance(
        TE[train_steps : train_steps + val_steps], args.P, args.Q
    )
    val_te = np.concatenate((val_te_x, val_te_y), axis=1)

    test_te_x, test_te_y = seq2instance(TE[-test_steps:], args.P, args.Q)
    test_te = np.concatenate((test_te_x, test_te_y), axis=1)

    return (
        trainX,
        train_te,
        trainY,
        valX,
        val_te,
        valY,
        testX,
        test_te,
        testY,
        SE,
        float(mean),
        float(std),
    )
