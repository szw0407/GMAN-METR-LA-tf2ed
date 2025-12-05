"""
GMAN Model Test Script - Optimized for Keras 3

Evaluates the trained GMAN model using modern Keras 3 APIs.
"""

import argparse
import logging
from pathlib import Path
from dataclasses import dataclass

import keras
import numpy as np
import tensorflow as tf

from model import GMAN, MaskedMAELoss
from utils import load_data, metric
from config import GMANConfig


@dataclass
class TestConfig:
    """Simple config for test.py"""
    time_slot: int = 5
    P: int = 12
    Q: int = 12
    L: int = 5
    K: int = 8
    d: int = 8
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    batch_size: int = 32
    max_epoch: int = 100
    patience: int = 10
    learning_rate: float = 0.001
    decay_epoch: int = 5
    traffic_file: str = "../data/METR-LA/metr-la.h5"
    SE_file: str = "../data/METR-LA/SE(METR).txt"
    model_file: str = "./models/GMAN.weights.h5"
    log_file: str = "./log/test_log"
    use_mixed_precision: bool = False


def main():
    """Test the GMAN model with Keras 3 optimized code."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test GMAN model")
    parser.add_argument('--time_slot', type=int, default=5, help='time interval (minutes)')
    parser.add_argument('--P', type=int, default=12, help='history steps')
    parser.add_argument('--Q', type=int, default=12, help='prediction steps')
    parser.add_argument('--L', type=int, default=5, help='number of STAtt Blocks')
    parser.add_argument('--K', type=int, default=8, help='number of attention heads')
    parser.add_argument('--d', type=int, default=8, help='dims of each head attention outputs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--traffic_file', default='../data/METR-LA/metr-la.h5', help='traffic file')
    parser.add_argument('--SE_file', default='../data/METR-LA/SE(METR).txt', help='spatial embedding file')
    parser.add_argument('--model_file', default='./models/GMAN.weights.h5', help='path to model weights')
    parser.add_argument('--log_file', default='./log/test_log', help='log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    Path("log").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.log_file, mode="w"),
            logging.StreamHandler()
        ],
    )
    log = logging.getLogger(__name__)
    
    log.info("=" * 70)
    log.info("GMAN Model Testing - Keras 3 Optimized")
    log.info("=" * 70)
    log.info(f"Arguments: {args}")
    
    # Check model file exists
    if not Path(args.model_file).exists():
        log.error(f"Model file not found: {args.model_file}")
        return
    
    # Configure GPU
    gpus = tf.config.list_physical_devices("GPU")
    log.info(f"GPU devices available: {len(gpus)}")
    
    # Create config for data loading
    try:
        config = TestConfig(
            time_slot=args.time_slot,
            P=args.P,
            Q=args.Q,
            L=args.L,
            K=args.K,
            d=args.d,
            batch_size=args.batch_size,
            traffic_file=args.traffic_file,
            SE_file=args.SE_file,
            model_file=args.model_file,
            log_file=args.log_file,
        )
        
        # Load data
        log.info("Loading data...")
        (
            trainX, trainTE, trainY,
            valX, valTE, valY,
            testX, testTE, testY,
            SE, mean, std
        ) = load_data(config)
        
        log.info(
            f"Data loaded successfully:\n"
            f"  trainX: {trainX.shape}, trainY: {trainY.shape}\n"
            f"  valX:   {valX.shape}, valY:   {valY.shape}\n"
            f"  testX:  {testX.shape}, testY:  {testY.shape}\n"
            f"  SE:     {SE.shape}"
        )
        
        # Create datasets - Keras 3 optimized
        def create_dataset(X, TE, Y):
            """Create optimized dataset for inference"""
            # Use deterministic dataset for inference to avoid caching warnings
            dataset = tf.data.Dataset.from_tensor_slices(((X, TE), Y))
            dataset = dataset.batch(args.batch_size, drop_remainder=False)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            # Disable caching for inference by using deterministic options
            options = tf.data.Options()
            options.experimental_deterministic = True
            dataset = dataset.with_options(options)
            return dataset
        
        train_ds = create_dataset(trainX, trainTE, trainY)
        val_ds = create_dataset(valX, valTE, valY)
        test_ds = create_dataset(testX, testTE, testY)
        
        # Build model
        log.info("Building model...")
        model = GMAN(config, SE, mean, std, bn=True)
        
        # Compile with modern Keras 3 API (using MaskedMAELoss)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=MaskedMAELoss(),
            metrics=[
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
                keras.metrics.MeanAbsolutePercentageError(name="mape"),
            ],
        )
        
        # Load weights
        log.info(f"Loading weights from {args.model_file}...")
        model.load_weights(args.model_file)
        log.info("Weights loaded successfully!")
        
        # Evaluate on all sets
        log.info("\n" + "=" * 70)
        log.info("Evaluating on all datasets...")
        log.info("=" * 70)
        
        # Training set
        log.info("\nTraining set evaluation:")
        train_pred = model.predict(train_ds, verbose="auto")
        train_mae, train_rmse, train_mape = metric(train_pred, trainY)
        
        # Validation set
        log.info("Validation set evaluation:")
        val_pred = model.predict(val_ds, verbose="auto")
        val_mae, val_rmse, val_mape = metric(val_pred, valY)
        
        # Test set
        log.info("Test set evaluation:")
        test_pred = model.predict(test_ds, verbose="auto")
        test_mae, test_rmse, test_mape = metric(test_pred, testY)
        
        # Summary
        log.info("\n" + "=" * 70)
        log.info("OVERALL RESULTS")
        log.info("=" * 70)
        log.info("                MAE\t\tRMSE\t\tMAPE")
        log.info(f"Train            {train_mae:.2f}\t\t{train_rmse:.2f}\t\t{train_mape * 100:.2f}%")
        log.info(f"Val              {val_mae:.2f}\t\t{val_rmse:.2f}\t\t{val_mape * 100:.2f}%")
        log.info(f"Test             {test_mae:.2f}\t\t{test_rmse:.2f}\t\t{test_mape * 100:.2f}%")
        
        # Per-step performance
        log.info("\n" + "=" * 70)
        log.info("PERFORMANCE FOR EACH PREDICTION STEP (Test Set)")
        log.info("=" * 70)
        log.info("Step\t\tMAE\t\tRMSE\t\tMAPE")
        
        mae_steps, rmse_steps, mape_steps = [], [], []
        for q in range(args.Q):
            mae, rmse, mape = metric(test_pred[:, q], testY[:, q])
            mae_steps.append(mae)
            rmse_steps.append(rmse)
            mape_steps.append(mape)
            log.info(f"{q + 1:02d}\t\t{mae:.2f}\t\t{rmse:.2f}\t\t{mape * 100:.2f}%")
        
        # Average
        avg_mae = np.mean(mae_steps)
        avg_rmse = np.mean(rmse_steps)
        avg_mape = np.mean(mape_steps)
        log.info("-" * 70)
        log.info(f"Average\t\t{avg_mae:.2f}\t\t{avg_rmse:.2f}\t\t{avg_mape * 100:.2f}%")
        
        log.info("\n" + "=" * 70)
        log.info("Testing completed successfully!")
        log.info(f"Results saved to: {args.log_file}")
        log.info("=" * 70)
        
    except Exception as e:
        log.error(f"Error during testing: {str(e)}")
        raise


if __name__ == '__main__':
    main()
