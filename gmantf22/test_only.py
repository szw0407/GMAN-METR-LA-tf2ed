"""
GMAN Model Test-Only Script - Optimized for Keras 3

Performs inference and evaluation on pre-trained GMAN model without training.
Supports evaluation on train/val/test sets with detailed metrics.
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
class TestOnlyConfig:
    """Simple config for test_only.py"""
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
    log_file: str = "./log/test_only_log"
    use_mixed_precision: bool = False


def main():
    """Test GMAN model on all datasets with detailed reporting."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test GMAN model on all datasets")
    parser.add_argument('--time_slot', type=int, default=5, help='time interval (minutes)')
    parser.add_argument('--P', type=int, default=12, help='history steps')
    parser.add_argument('--Q', type=int, default=12, help='prediction steps')
    parser.add_argument('--L', type=int, default=5, help='number of STAtt Blocks')
    parser.add_argument('--K', type=int, default=8, help='number of attention heads')
    parser.add_argument('--d', type=int, default=8, help='dims of each head attention outputs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--traffic_file', default='../data/METR-LA/metr-la.h5', help='traffic file')
    parser.add_argument('--SE_file', default='../data/METR-LA/SE(METR).txt', help='spatial embedding file')
    parser.add_argument('--model_file', default='./models/GMAN.weights.h5', help='saved model path')
    parser.add_argument('--log_file', default='./log/test_only_log', help='log file')
    
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
    
    log.info("=" * 80)
    log.info("GMAN Model Test-Only (Inference) - Keras 3 Optimized")
    log.info("=" * 80)
    log.info(f"Arguments: {args}")
    
    # Check model file
    if not Path(args.model_file).exists():
        log.error(f"Model file not found: {args.model_file}")
        return
    
    # Configure GPU for inference
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if len(gpus) > 0:
        log.info(f"GPU inference enabled: {len(gpus)} device(s)")
    else:
        log.info("Using CPU for inference")
    
    try:
        # Create config for data loading
        config = TestOnlyConfig(
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
        log.info("\nLoading data...")
        (
            trainX, trainTE, trainY,
            valX, valTE, valY,
            testX, testTE, testY,
            SE, mean, std
        ) = load_data(config)
        
        log.info(
            f"Data loaded:\n"
            f"  trainX: {trainX.shape}, trainY: {trainY.shape}\n"
            f"  valX:   {valX.shape}, valY:   {valY.shape}\n"
            f"  testX:  {testX.shape}, testY:  {testY.shape}\n"
            f"  SE:     {SE.shape}\n"
            f"  Normalization - Mean: {mean:.4f}, Std: {std:.4f}"
        )
        
        # Create datasets - Keras 3 optimized for inference
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
        
        log.info("tf.data.Dataset pipelines created")
        
        # Build model
        log.info("\nBuilding model...")
        model = GMAN(config, SE, mean, std, bn=True)
        
        # Compile for inference (Keras 3 optimized)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=MaskedMAELoss(),
            metrics=[
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
                keras.metrics.MeanAbsolutePercentageError(name="mape"),
            ],
        )
        
        log.info("Model compiled successfully")
        
        # Load pre-trained weights
        log.info(f"\nLoading model weights from {args.model_file}...")
        model.load_weights(args.model_file)
        log.info("Model weights loaded!")
        
        # Perform inference on all sets
        log.info("\n" + "=" * 80)
        log.info("INFERENCE ON ALL DATASETS")
        log.info("=" * 80)
        
        # Training set inference
        log.info("\n>>> Evaluating on TRAINING SET...")
        train_pred = model.predict(train_ds, verbose="auto")
        train_mae, train_rmse, train_mape = metric(train_pred, trainY)
        
        # Validation set inference
        log.info("\n>>> Evaluating on VALIDATION SET...")
        val_pred = model.predict(val_ds, verbose="auto")
        val_mae, val_rmse, val_mape = metric(val_pred, valY)
        
        # Test set inference
        log.info("\n>>> Evaluating on TEST SET...")
        test_pred = model.predict(test_ds, verbose="auto")
        test_mae, test_rmse, test_mape = metric(test_pred, testY)
        
        # Overall summary
        log.info("\n" + "=" * 80)
        log.info("OVERALL PERFORMANCE SUMMARY")
        log.info("=" * 80)
        log.info(f"\n{'Dataset':<15} {'MAE':<12} {'RMSE':<12} {'MAPE':<12}")
        log.info("-" * 80)
        log.info(f"{'Train':<15} {train_mae:<12.4f} {train_rmse:<12.4f} {train_mape * 100:<12.2f}%")
        log.info(f"{'Validation':<15} {val_mae:<12.4f} {val_rmse:<12.4f} {val_mape * 100:<12.2f}%")
        log.info(f"{'Test':<15} {test_mae:<12.4f} {test_rmse:<12.4f} {test_mape * 100:<12.2f}%")
        
        # Detailed per-step performance on test set
        log.info("\n" + "=" * 80)
        log.info("DETAILED PER-STEP PERFORMANCE (TEST SET)")
        log.info("=" * 80)
        log.info(f"\n{'Step':<8} {'MAE':<12} {'RMSE':<12} {'MAPE':<12}")
        log.info("-" * 80)
        
        mae_steps, rmse_steps, mape_steps = [], [], []
        for q in range(args.Q):
            mae, rmse, mape = metric(test_pred[:, q], testY[:, q])
            mae_steps.append(mae)
            rmse_steps.append(rmse)
            mape_steps.append(mape)
            log.info(f"{q + 1:<8} {mae:<12.4f} {rmse:<12.4f} {mape * 100:<12.2f}%")
        
        # Average per-step
        avg_mae = np.mean(mae_steps)
        avg_rmse = np.mean(rmse_steps)
        avg_mape = np.mean(mape_steps)
        log.info("-" * 80)
        log.info(f"{'AVERAGE':<8} {avg_mae:<12.4f} {avg_rmse:<12.4f} {avg_mape * 100:<12.2f}%")
        
        # Model architecture info
        log.info("\n" + "=" * 80)
        log.info("MODEL ARCHITECTURE")
        log.info("=" * 80)
        log.info(f"Model parameters:")
        log.info(f"  - Time slots per day (T): {24 * 60 // args.time_slot}")
        log.info(f"  - History steps (P): {args.P}")
        log.info(f"  - Prediction steps (Q): {args.Q}")
        log.info(f"  - Attention blocks (L): {args.L}")
        log.info(f"  - Number of heads (K): {args.K}")
        log.info(f"  - Head dimension (d): {args.d}")
        log.info(f"  - Model dimension (D=K*d): {args.K * args.d}")
        log.info(f"  - Number of sensors (N): {SE.shape[0]}")
        
        # Final summary
        log.info("\n" + "=" * 80)
        log.info("TEST-ONLY EVALUATION COMPLETED SUCCESSFULLY!")
        log.info("=" * 80)
        log.info(f"Results saved to: {args.log_file}")
        log.info("=" * 80)
        
    except Exception as e:
        log.error(f"Error during inference: {str(e)}")
        import traceback
        log.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()