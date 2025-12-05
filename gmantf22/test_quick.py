"""
Test Script for Quick-Trained GMAN Model - Unified Version

Comprehensive evaluation of quick-trained GMAN models with detailed metrics and visualization.
Optimized for rapid iteration and detailed performance assessment.

Usage:
    python test_quick.py                          # Test quick model (default)
    python test_quick.py --config full            # Test full model
    python test_quick.py --model ./models/GMAN.weights.h5  # Custom model path
    python test_quick.py --output-dir ./results   # Custom output directory
"""

import argparse
import logging
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import GMAN, MaskedMAELoss
from utils import load_data, metric
from config_quick import GMANConfigQuick
from config import GMANConfig


def setup_logging(log_file: str):
    """Setup logging to file and console"""
    Path("log").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def create_dataset(X, TE, Y, batch_size: int, use_mixed_precision: bool):
    """Create optimized dataset for inference"""
    compute_dtype = tf.float16 if use_mixed_precision else tf.float32
    
    ds = tf.data.Dataset.from_tensor_slices(((X, TE), Y))
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.map(
        lambda features, label: (
            (tf.cast(features[0], compute_dtype), features[1]),
            tf.cast(label, compute_dtype),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def plot_predictions(y_pred: np.ndarray, y_true: np.ndarray, output_path: str, num_samples: int = 3):
    """Plot sample predictions vs ground truth with error analysis"""
    num_samples = min(num_samples, y_pred.shape[0])
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 3 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Time series comparison
        axes[i, 0].plot(y_true[i, :, 0], label="Ground Truth", linewidth=2, marker='o', markersize=5)
        axes[i, 0].plot(y_pred[i, :, 0], label="Prediction", linewidth=2, marker='s', alpha=0.7, markersize=5)
        axes[i, 0].set_title(f"Sample {i+1}: Predictions (Node 0)", fontweight="bold")
        axes[i, 0].set_xlabel("Time Step")
        axes[i, 0].set_ylabel("Traffic Flow")
        axes[i, 0].legend(loc='best')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Error distribution
        error = np.abs(y_pred[i] - y_true[i]).flatten()
        axes[i, 1].hist(error, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[i, 1].set_title(f"Sample {i+1}: Absolute Error Distribution", fontweight="bold")
        axes[i, 1].set_xlabel("Absolute Error")
        axes[i, 1].set_ylabel("Frequency")
        axes[i, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Unified GMAN quick model test with comprehensive evaluation and visualization"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Test Quick-Trained GMAN Model - Keras 3 Optimized",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_quick.py                          # Test quick model (L=2, K=4, d=4)
  python test_quick.py --config full            # Test full model (L=5, K=8, d=8)
  python test_quick.py --model ./models/GMAN.weights.h5  # Custom model
  python test_quick.py --output-dir ./results   # Custom output directory
        """
    )
    
    parser.add_argument('--config', type=str, default='quick', choices=['quick', 'full'],
                        help='Configuration: quick (simplified) or full (complete)')
    parser.add_argument('--model', type=str, default='./models/GMAN_quick.weights.h5',
                        help='Path to model weights file')
    parser.add_argument('--output-dir', type=str, default='./test_results',
                        help='Output directory for results and plots')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size (default: use config value)')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log = setup_logging(str(output_dir / "test_report.log"))
    
    log.info("=" * 80)
    log.info("GMAN Model Unified Quick Test - Keras 3 Optimized")
    log.info("=" * 80)
    log.info(f"Configuration: {args.config.upper()}")
    log.info(f"Model weights: {args.model}")
    log.info(f"Output directory: {output_dir}")
    
    # Verify model exists
    if not Path(args.model).exists():
        log.error(f"Model file not found: {args.model}")
        return
    
    # Setup GPU
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    log.info(f"GPU devices available: {len(gpus)}")
    
    try:
        # Load config
        if args.config == "quick":
            config = GMANConfigQuick()
            log.info("Using QUICK config: L=2, K=4, d=4, batch_size=84")
        else:
            config = GMANConfig()
            log.info("Using FULL config: L=5, K=8, d=8, batch_size=12")
        
        # Override batch size if provided
        if args.batch_size:
            config.batch_size = args.batch_size
            log.info(f"Batch size overridden to: {args.batch_size}")
        
        # Load data
        log.info("\n" + "=" * 80)
        log.info("Loading data...")
        log.info("=" * 80)
        
        (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE, mean, std) = load_data(config)
        
        log.info(f"Data loaded successfully:")
        log.info(f"  trainX: {trainX.shape}, trainY: {trainY.shape}")
        log.info(f"  valX:   {valX.shape}, valY:   {valY.shape}")
        log.info(f"  testX:  {testX.shape}, testY:  {testY.shape}")
        log.info(f"  SE:     {SE.shape}")
        log.info(f"  Normalization: mean={mean:.4f}, std={std:.4f}")
        
        # Create datasets
        log.info("\nPreparing datasets...")
        train_ds = create_dataset(trainX, trainTE, trainY, config.batch_size, config.use_mixed_precision)
        val_ds = create_dataset(valX, valTE, valY, config.batch_size, config.use_mixed_precision)
        test_ds = create_dataset(testX, testTE, testY, config.batch_size, config.use_mixed_precision)
        log.info(f"Datasets prepared (batch_size={config.batch_size})")
        
        # Build and compile model
        log.info("\n" + "=" * 80)
        log.info("Building and loading model...")
        log.info("=" * 80)
        
        model = GMAN(config, SE, mean, std, bn=False if args.config == "quick" else True)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=MaskedMAELoss(),
                     metrics=[keras.metrics.MeanAbsoluteError(name="mae"),
                             keras.metrics.RootMeanSquaredError(name="rmse"),
                             keras.metrics.MeanAbsolutePercentageError(name="mape")])
        log.info("Model compiled")
        
        # Build model by running on sample batch
        log.info("Building model...")
        sample_x = (tf.constant(trainX[:config.batch_size], dtype=tf.float32),
                   tf.constant(trainTE[:config.batch_size], dtype=tf.int32))
        _ = model(sample_x, training=False)
        log.info("Model built successfully")
        
        # Load weights
        log.info(f"Loading weights from: {args.model}")
        model.load_weights(args.model)
        log.info("Model weights loaded!")
        
        # Inference on all sets
        log.info("\n" + "=" * 80)
        log.info("Running inference on all datasets...")
        log.info("=" * 80)
        
        log.info("\n>> Training set...")
        train_pred = model.predict(train_ds, verbose=0)
        train_mae, train_rmse, train_mape = metric(train_pred, trainY)
        
        log.info(">> Validation set...")
        val_pred = model.predict(val_ds, verbose=0)
        val_mae, val_rmse, val_mape = metric(val_pred, valY)
        
        log.info(">> Test set...")
        test_pred = model.predict(test_ds, verbose=0)
        test_mae, test_rmse, test_mape = metric(test_pred, testY)
        
        # Overall summary
        log.info("\n" + "=" * 80)
        log.info("OVERALL PERFORMANCE SUMMARY")
        log.info("=" * 80)
        log.info(f"{'Dataset':<15} {'MAE':<12} {'RMSE':<12} {'MAPE':<10}")
        log.info("-" * 50)
        log.info(f"{'Train':<15} {train_mae:<12.4f} {train_rmse:<12.4f} {train_mape * 100:<10.2f}%")
        log.info(f"{'Validation':<15} {val_mae:<12.4f} {val_rmse:<12.4f} {val_mape * 100:<10.2f}%")
        log.info(f"{'Test':<15} {test_mae:<12.4f} {test_rmse:<12.4f} {test_mape * 100:<10.2f}%")
        
        # Per-step performance
        log.info("\n" + "=" * 80)
        log.info("PER-STEP PERFORMANCE (Test Set)")
        log.info("=" * 80)
        log.info(f"{'Step':<8} {'MAE':<12} {'RMSE':<12} {'MAPE':<10}")
        log.info("-" * 50)
        
        mae_steps, rmse_steps, mape_steps = [], [], []
        for q in range(config.Q):
            mae, rmse, mape = metric(test_pred[:, q], testY[:, q])
            mae_steps.append(mae)
            rmse_steps.append(rmse)
            mape_steps.append(mape)
            log.info(f"{q + 1:<8} {mae:<12.4f} {rmse:<12.4f} {mape * 100:<10.2f}%")
        
        # Average
        avg_mae = np.mean(mae_steps)
        avg_rmse = np.mean(rmse_steps)
        avg_mape = np.mean(mape_steps)
        log.info("-" * 50)
        log.info(f"{'AVERAGE':<8} {avg_mae:<12.4f} {avg_rmse:<12.4f} {avg_mape * 100:<10.2f}%")
        
        # Model architecture info
        log.info("\n" + "=" * 80)
        log.info("MODEL ARCHITECTURE")
        log.info("=" * 80)
        log.info(f"Time slots per day (T): {24 * 60 // config.time_slot}")
        log.info(f"History steps (P): {config.P}")
        log.info(f"Prediction steps (Q): {config.Q}")
        log.info(f"Attention blocks (L): {config.L}")
        log.info(f"Number of heads (K): {config.K}")
        log.info(f"Head dimension (d): {config.d}")
        log.info(f"Model dimension (D=K*d): {config.K * config.d}")
        log.info(f"Number of sensors (N): {SE.shape[0]}")
        
        # Generate visualizations
        log.info("\n" + "=" * 80)
        log.info("Generating visualizations...")
        log.info("=" * 80)
        
        plot_predictions(trainY, train_pred, str(output_dir / "train_predictions.png"), num_samples=3)
        log.info(f"  Saved: {output_dir / 'train_predictions.png'}")
        
        plot_predictions(valY, val_pred, str(output_dir / "val_predictions.png"), num_samples=3)
        log.info(f"  Saved: {output_dir / 'val_predictions.png'}")
        
        plot_predictions(testY, test_pred, str(output_dir / "test_predictions.png"), num_samples=3)
        log.info(f"  Saved: {output_dir / 'test_predictions.png'}")
        
        # Final summary
        log.info("\n" + "=" * 80)
        log.info("TESTING COMPLETED SUCCESSFULLY!")
        log.info("=" * 80)
        log.info(f"Results saved to: {output_dir}")
        log.info(f"Log file: {output_dir / 'test_report.log'}")
        log.info("=" * 80)
        
    except Exception as e:
        log.error(f"Error during testing: {str(e)}")
        import traceback
        log.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
