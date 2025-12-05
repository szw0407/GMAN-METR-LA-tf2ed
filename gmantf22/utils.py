import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
# keras.mixed_precision.set_global_policy("mixed_float16")
from config import GMANConfig

def log_string(log: logging.Logger, string: str):
    """
    Write a string to the log file and print it to the console.
    """
    log.info(string)
    print(string)


def metric(pred: np.ndarray, label: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate performance metrics: MAE, RMSE, MAPE using TensorFlow ops (GPU-accelerated).
    This function replicates the original's masking behavior for zero values in labels.
    
    Args:
        pred: Prediction array (can be numpy or tensor)
        label: Ground truth array (can be numpy or tensor)
    
    Returns:
        Tuple of (MAE, RMSE, MAPE) as Python floats
    """
    # Convert to tensors on GPU with float32 precision
    pred = tf.cast(pred, tf.float32)
    label = tf.cast(label, tf.float32)

    # Create mask for non-zero labels (GPU operation)
    mask = tf.not_equal(label, 0.0)

    # If all labels are zero, metrics are undefined and should be 0
    if not tf.reduce_any(mask):
        return 0.0, 0.0, 0.0

    # Filter out the zero-valued data points using mask (GPU operation)
    pred_masked = tf.boolean_mask(pred, mask)
    label_masked = tf.boolean_mask(label, mask)

    # Calculate MAE using TensorFlow (GPU-accelerated)
    mae = tf.reduce_mean(tf.abs(label_masked - pred_masked))
    
    # Calculate RMSE using TensorFlow (GPU-accelerated)
    mse = tf.reduce_mean(tf.square(label_masked - pred_masked))
    rmse = tf.sqrt(mse)

    # MAPE calculation with epsilon to prevent division by zero and clipping to prevent overflow
    # Add epsilon to denominator to avoid division by very small numbers
    epsilon = 1e-3  # 防止除以接近零的流量值
    mape = tf.reduce_mean(tf.abs((label_masked - pred_masked) / tf.maximum(label_masked, epsilon)))
    # Clip MAPE to reasonable range (0-2 means 0-200%) to prevent overflow in mixed precision
    mape = tf.clip_by_value(mape, 0.0, 2.0)

    # Convert results to Python floats
    return float(mae.numpy()), float(rmse.numpy()), float(mape.numpy())


def seq2instance(data: np.ndarray, P: int, Q: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert sequence data into instances for training/testing using TensorFlow strided_slice (GPU-optimized).
    
    This implementation uses TensorFlow operations to create sliding windows efficiently,
    avoiding explicit Python loops that would be executed on CPU.
    
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

    # Convert to TensorFlow constant for GPU operations
    data_tensor = tf.constant(data, dtype=tf.float32)
    
    # Create index arrays for efficient slicing
    # Each sample i will have indices [i, i+1, ..., i+P+Q-1]
    indices = tf.range(num_sample)[:, tf.newaxis] + tf.range(P + Q)[tf.newaxis, :]
    
    # Use gather to extract all windows at once (GPU operation)
    windows = tf.gather(data_tensor, indices)  # shape: (num_sample, P+Q, dims)
    
    # Split into x and y
    x = windows[:, :P, :]      # shape: (num_sample, P, dims)
    y = windows[:, P:, :]      # shape: (num_sample, Q, dims)
    
    # Convert to numpy arrays
    return x.numpy(), y.numpy()


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

    # Normalize features using TensorFlow operations (GPU-optimized)
    trainX_tensor = tf.constant(trainX, dtype=tf.float32)
    mean = tf.reduce_mean(trainX_tensor)
    std = tf.math.reduce_std(trainX_tensor)
    std = tf.maximum(std, 1e-5)  # Avoid division by zero
    
    # Apply normalization using TensorFlow (GPU operation)
    trainX = ((tf.constant(trainX, dtype=tf.float32) - mean) / std).numpy()
    valX = ((tf.constant(valX, dtype=tf.float32) - mean) / std).numpy()
    testX = ((tf.constant(testX, dtype=tf.float32) - mean) / std).numpy()
    
    mean = float(mean.numpy())
    std = float(std.numpy())

    # Load and normalize spatial embeddings
    try:
        # Use np.loadtxt for efficient parsing of the text file (required for file I/O)
        SE = np.loadtxt(se_path, skiprows=1)[:, 1:].astype(np.float32)

        # Normalize SE using TensorFlow (GPU-optimized)
        SE_tensor = tf.constant(SE, dtype=tf.float32)
        se_mean = tf.reduce_mean(SE_tensor)
        se_std = tf.math.reduce_std(SE_tensor)
        se_std = tf.maximum(se_std, 1e-5)
        
        SE = ((SE_tensor - se_mean) / se_std).numpy()
        se_mean = float(se_mean.numpy())
        se_std = float(se_std.numpy())

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
