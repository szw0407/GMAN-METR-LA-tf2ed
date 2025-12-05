"""
GMAN Model Training Script - Optimized for Keras 3

This script trains the GMAN model using modern Keras 3 APIs with:
- Native Keras 3 training loop
- Automatic mixed precision training
- GPU memory growth for stability
- TensorBoard integration
- Modern callbacks and metrics
"""

import logging
from pathlib import Path
import datetime

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import GMAN, MaskedMAELoss
from utils import load_data, metric
from config import GMANConfig

# Initialize configuration
args = GMANConfig()  # type: ignore
# keras.mixed_precision.set_global_policy("mixed_float16") if args.use_mixed_precision else keras.mixed_precision.set_global_policy("float32")

def setup_environment(log_file: str, use_mixed_precision: bool):
    """
    Configures the runtime environment: logging, GPU, and mixed precision.
    Optimized for Keras 3.
    """
    # Setup logging
    Path("log").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)
    log.info("Logging setup complete.")
    log.info(str(args))

    # Configure GPU devices with memory growth for stability
    gpus = tf.config.list_physical_devices("GPU")
    log.info(f"Detected {len(gpus)} GPU device(s).")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        log.info(f"  - {gpu.name}: Memory growth enabled.")

    # Configure mixed precision (Keras 3 automatically handles this)
    if use_mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
        log.info("Mixed precision (float16) enabled.")
    else:
        log.info("Using default precision (float32).")
    
    return log


def prepare_datasets(
    trainX: np.ndarray,
    trainTE: np.ndarray,
    trainY: np.ndarray,
    valX: np.ndarray,
    valTE: np.ndarray,
    valY: np.ndarray,
    testX: np.ndarray,
    testTE: np.ndarray,
    testY: np.ndarray,
    batch_size: int,
    log: logging.Logger,
):
    """
    Creates tf.data.Dataset pipelines for training, validation, and testing.
    Optimized for GPU performance with prefetching, caching, and parallel processing.
    """

    def create_dataset(X, TE, Y, shuffle=False, cache_data=False):
        # Convert to TensorFlow constants on GPU
        ds = tf.data.Dataset.from_tensor_slices(((X, TE), Y))
        
        # Shuffle training data
        if shuffle:
            ds = ds.shuffle(buffer_size=min(2048, len(X)))
        
        # Batch the data
        ds = ds.batch(batch_size, drop_remainder=False)
        
        # Cache for training data (will be read multiple times per epoch)
        # For val/test, caching is not necessary
        if cache_data:
            ds = ds.cache()
        
        # Parallel prefetching for GPU pipeline optimization
        # AUTOTUNE automatically selects the number of prefetch threads
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds

    train_ds = create_dataset(trainX, trainTE, trainY, shuffle=True, cache_data=True)
    val_ds = create_dataset(valX, valTE, valY, cache_data=False)
    test_ds = create_dataset(testX, testTE, testY, cache_data=False)

    log.info("tf.data.Dataset pipelines created and GPU-optimized:")
    log.info("  - Shuffle buffer enabled for training")
    log.info("  - Cache layer enabled for training data (read per epoch)")
    log.info("  - AUTOTUNE prefetching enabled for GPU pipeline optimization")
    
    return train_ds, val_ds, test_ds


def build_and_train_model(
    log: logging.Logger,
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    max_epoch: int,
    patience: int,
    model_file: str,
    learning_rate: float,
    decay_epoch: int,
    steps_per_epoch: int,
):
    """
    Compiles and trains the GMAN model using modern Keras 3 APIs.
    Includes automatic gradient scaling for mixed precision.
    """
    log.info("**** Building and training model ****")

    # Create learning rate schedule - Keras 3 standard
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_epoch * steps_per_epoch,
        decay_rate=0.7,
        staircase=True,
    )

    # Compile with modern Keras 3 optimizer configuration
    # Global clip norm is applied automatically by Keras 3
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=lr_schedule,  # type:ignore
            global_clipnorm=5.0,  # Keras 3 feature: automatic gradient clipping
        ),  # type: ignore
        loss=MaskedMAELoss(),
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
            keras.metrics.MeanAbsolutePercentageError(name="mape"),
        ],
    )

    # TensorBoard callback with modern Keras 3 support
    log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    callbacks = [
        # Early stopping with best weight restoration
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        # Model checkpoint - save best weights
        keras.callbacks.ModelCheckpoint(
            filepath=model_file,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.7,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch",
        ),
    ]

    # Train model - Keras 3 handles mixed precision automatically
    log.info(f"Training for {max_epoch} epochs...")
    history = model.fit(
        train_ds,
        epochs=max_epoch,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose='auto',
    )

    log.info("Model training complete.")

    # Visualize training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    axes[0].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_title("Loss during Training", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Metrics curve
    if "mae" in history.history:
        axes[1].plot(history.history["mae"], label="Train MAE", linewidth=2)
        axes[1].plot(history.history["val_mae"], label="Val MAE", linewidth=2)
    if "mape" in history.history:
        axes[1].plot(history.history["mape"], label="Train MAPE", linewidth=2)
        axes[1].plot(history.history["val_mape"], label="Val MAPE", linewidth=2)
    
    axes[1].set_title("Metrics during Training", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = f"{log_dir}/training_curves.png"
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    log.info(f"Training curves saved to {fig_path}")

    return history, log_dir


def evaluate_model(
    log: logging.Logger,
    model: keras.Model,
    test_ds: tf.data.Dataset,
    testY: np.ndarray,
    Q: int,
    model_file: str,
):
    """
    Evaluates the model on the test set and logs the performance.
    Keras 3 optimized evaluation with GPU-accelerated metrics.
    """
    log.info("**** Testing model ****")
    
    # Load best weights
    model.load_weights(model_file)
    log.info("Best model weights restored for testing.")

    # Perform prediction once (Keras 3 handles batch processing)
    test_pred = model.predict(test_ds, verbose=0)

    # Overall performance using TensorFlow-accelerated metric function
    test_mae, test_rmse, test_mape = metric(test_pred, testY)
    log.info("                MAE\t\tRMSE\t\tMAPE")
    log.info(
        f"Overall Test     {test_mae:.2f}\t\t{test_rmse:.2f}\t\t{test_mape * 100:.2f}%"
    )

    # Per-step performance using TensorFlow operations (GPU-accelerated)
    log.info("Performance for each prediction step:")
    
    # Convert to tensors for batch GPU processing
    test_pred_tensor = tf.constant(test_pred, dtype=tf.float32)
    testY_tensor = tf.constant(testY, dtype=tf.float32)
    
    # Extract per-step metrics using TensorFlow slicing (GPU operation)
    mae_steps = []
    rmse_steps = []
    mape_steps = []
    
    for q in range(Q):
        mae, rmse, mape = metric(test_pred_tensor[:, q].numpy(), testY_tensor[:, q].numpy())
        mae_steps.append(mae)
        rmse_steps.append(rmse)
        mape_steps.append(mape)
        log.info(
            f"  - Step {q + 1:02d}:    {mae:.2f}\t\t{rmse:.2f}\t\t{mape * 100:.2f}%"
        )

    # Average per-step performance using TensorFlow (GPU-accelerated)
    avg_mae = float(tf.reduce_mean(tf.constant(mae_steps, dtype=tf.float32)).numpy())
    avg_rmse = float(tf.reduce_mean(tf.constant(rmse_steps, dtype=tf.float32)).numpy())
    avg_mape = float(tf.reduce_mean(tf.constant(mape_steps, dtype=tf.float32)).numpy())
    log.info(
        f"Average Steps:   {avg_mae:.2f}\t\t{avg_rmse:.2f}\t\t{avg_mape * 100:.2f}%"
    )


def main():
    """
    Main function to run the GMAN model training pipeline with Keras 3 optimizations.
    """
    Path("models").mkdir(exist_ok=True)
    log = setup_environment(args.log_file, args.use_mixed_precision)

    # Load data
    log.info("Loading data...")
    (
        trainX,
        trainTE,
        trainY,
        valX,
        valTE,
        valY,
        testX,
        testTE,
        testY,
        SE,
        mean,
        std,
    ) = load_data(args)
    log.info(
        f"Data loaded. Shapes:\n"
        f"  trainX: {trainX.shape}, trainY: {trainY.shape}\n"
        f"  valX:   {valX.shape}, valY:   {valY.shape}\n"
        f"  testX:  {testX.shape}, testY:  {testY.shape}\n"
        f"  SE:     {SE.shape}\n"
        f"  Mean: {mean:.2f}, Std: {std:.2f}"
    )

    # Prepare datasets
    train_ds, val_ds, test_ds = prepare_datasets(
        trainX,
        trainTE,
        trainY,
        valX,
        valTE,
        valY,
        testX,
        testTE,
        testY,
        args.batch_size,
        log,
    )

    # Build model
    log.info("Building GMAN model...")
    model = GMAN(args, SE, mean, std, bn=True)
    
    # Calculate steps per epoch for learning rate schedule
    steps_per_epoch = len(trainX) // args.batch_size

    # Train and evaluate
    history, log_dir = build_and_train_model(
        log,
        model,
        train_ds,
        val_ds,
        args.max_epoch,
        args.patience,
        args.model_file,
        args.learning_rate,
        args.decay_epoch,
        steps_per_epoch,
    )

    # Log model summary
    log.info("Model Architecture Summary:")
    model.summary(print_fn=lambda x: log.info(x))

    # Evaluate on test set
    evaluate_model(log, model, test_ds, testY, args.Q, args.model_file)

    log.info("Training and evaluation completed successfully!")
    log.info(f"Results logged to: {log_dir}")


if __name__ == "__main__":
    main()
