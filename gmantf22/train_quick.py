"""
快速训练脚本 - 用于快速得到一个可行的模型
配置相比完整版简化了模型复杂度，但保持了核心结构
"""

import logging
from pathlib import Path
import datetime

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import GMAN, MaskedMAELoss, MaskedMAE, MaskedRMSE, MaskedMAPE
from utils import load_data, metric
from config_quick import GMANConfigQuick

# Initialize configuration
args = GMANConfigQuick()

def setup_environment(log_file: str, use_mixed_precision: bool):
    """配置运行环境"""
    Path("log").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)
    log.info("=" * 80)
    log.info("QUICK TRAINING MODE - Simplified model for rapid iteration")
    log.info("=" * 80)
    log.info("GMAN MODEL CONFIGURATION (QUICK MODE)")
    log.info("=" * 80)
    log.info(f"Model Architecture:")
    log.info(f"  - L (STAtt Blocks): {args.L}")
    log.info(f"  - K (Attention Heads): {args.K}")
    log.info(f"  - d (Head Output Dims): {args.d}")
    log.info(f"Temporal Configuration:")
    log.info(f"  - P (History Steps): {args.P}")
    log.info(f"  - Q (Prediction Steps): {args.Q}")
    log.info(f"  - Time Slot (minutes): {args.time_slot}")
    log.info(f"Training Configuration:")
    log.info(f"  - Batch Size: {args.batch_size}")
    log.info(f"  - Learning Rate: {args.learning_rate}")
    log.info(f"  - Decay Epoch: {args.decay_epoch}")
    log.info(f"  - Max Epoch: {args.max_epoch}")
    log.info(f"  - Patience (Early Stop): {args.patience}")
    log.info(f"Data Configuration:")
    log.info(f"  - Train/Val/Test Split: {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")
    log.info(f"  - Traffic File: {args.traffic_file}")
    log.info(f"  - SE File: {args.SE_file}")
    log.info(f"Optimization Flags:")
    log.info(f"  - Mixed Precision: {args.use_mixed_precision}")
    log.info(f"  - XLA JIT: {args.enable_xla}")
    log.info(f"Output Files:")
    log.info(f"  - Model File: {args.model_file}")
    log.info(f"  - Log File: {args.log_file}")
    log.info("=" * 80)

    gpus = tf.config.list_physical_devices("GPU")
    log.info(f"Detected {len(gpus)} GPU device(s).")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # 快速模式禁用 XLA
    tf.config.optimizer.set_jit(False)
    log.info("XLA JIT disabled for faster compilation.")

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
    use_mixed_precision: bool,
    log: logging.Logger,
):
    """创建数据管线"""
    compute_dtype = tf.float16 if use_mixed_precision else tf.float32

    def create_dataset(X, TE, Y, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices(((X, TE), Y))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(2 * batch_size, len(X)))
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.map(
            lambda features, label: (
                (tf.cast(features[0], compute_dtype), features[1]),
                tf.cast(label, compute_dtype),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = create_dataset(trainX, trainTE, trainY, shuffle=True)
    val_ds = create_dataset(valX, valTE, valY, shuffle=False)
    test_ds = create_dataset(testX, testTE, testY, shuffle=False)

    log.info("Dataset pipelines created")
    log.info(f"  - Batch size: {batch_size}")
    log.info(f"  - Training batches: {len(train_ds)}")
    log.info(f"  - Validation batches: {len(val_ds)}")
    
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
    """构建和训练模型"""
    log.info("**** Building and training model (QUICK MODE) ****")

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_epoch * steps_per_epoch,
        decay_rate=0.7,
        staircase=True,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=lr_schedule,
            global_clipnorm=5.0,
        ),
        loss=MaskedMAELoss(),
        metrics=[
            MaskedMAE(),
            MaskedRMSE(),
            MaskedMAPE(),
        ],
    )

    log_dir = f"logs/fit_quick/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    log.info(f"TensorBoard logs: {log_dir}")
    log.info(f"Command: tensorboard --logdir=logs/fit_quick")
    tf.debugging.experimental.enable_dump_debug_info(log_dir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

    # Create a summary writer for hyperparameter logging
    summary_writer = tf.summary.create_file_writer(log_dir)

    # Log all hyperparameters from config to TensorBoard
    with summary_writer.as_default():
        # Model architecture hyperparameters
        tf.summary.scalar('hparams/L_num_statt_blocks', args.L, step=0)
        tf.summary.scalar('hparams/K_num_attention_heads', args.K, step=0)
        tf.summary.scalar('hparams/d_head_output_dims', args.d, step=0)
        
        # Temporal hyperparameters
        tf.summary.scalar('hparams/P_history_steps', args.P, step=0)
        tf.summary.scalar('hparams/Q_prediction_steps', args.Q, step=0)
        tf.summary.scalar('hparams/time_slot_minutes', args.time_slot, step=0)
        
        # Training hyperparameters
        tf.summary.scalar('hparams/batch_size', args.batch_size, step=0)
        tf.summary.scalar('hparams/learning_rate', args.learning_rate, step=0)
        tf.summary.scalar('hparams/decay_epoch', args.decay_epoch, step=0)
        tf.summary.scalar('hparams/max_epoch', args.max_epoch, step=0)
        tf.summary.scalar('hparams/patience_early_stop', args.patience, step=0)
        
        # Data split ratios
        tf.summary.scalar('hparams/train_ratio', args.train_ratio, step=0)
        tf.summary.scalar('hparams/val_ratio', args.val_ratio, step=0)
        tf.summary.scalar('hparams/test_ratio', args.test_ratio, step=0)
        
        # Optimization flags
        tf.summary.scalar('hparams/use_mixed_precision', float(args.use_mixed_precision), step=0)
        tf.summary.scalar('hparams/enable_xla', float(args.enable_xla), step=0)
    
    log.info("Hyperparameters logged to TensorBoard")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_file,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        # Note: Learning rate is controlled by ExponentialDecay schedule, not ReduceLROnPlateau
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,        # Log weight histograms every epoch
            write_graph=True,        # Write model graph
            write_images=True,       # Visualize model weights as images
            update_freq='epoch',     # Update metrics every epoch
            profile_batch="5,10",    # Profile batches 5-10 for performance analysis
            embeddings_freq=1,       # Log embeddings every epoch
        ),
    ]

    log.info(f"Training for {max_epoch} epochs (QUICK MODE)...")
    history = model.fit(
        train_ds,
        epochs=max_epoch,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose='auto',
    )

    log.info("Quick training complete!")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_title("Loss - Quick Training", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "masked_mae" in history.history:
        axes[1].plot(history.history["masked_mae"], label="Train MAE", linewidth=2)
        axes[1].plot(history.history["val_masked_mae"], label="Val MAE", linewidth=2)
    
    axes[1].set_title("MAE - Quick Training", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
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
    """评估模型"""
    log.info("**** Testing model ****")
    
    model.load_weights(model_file)
    log.info("Best model weights loaded.")

    test_pred = model.predict(test_ds, verbose=0)

    test_mae, test_rmse, test_mape = metric(test_pred, testY)
    log.info("                MAE\t\tRMSE\t\tMAPE")
    log.info(
        f"Overall Test     {test_mae:.2f}\t\t{test_rmse:.2f}\t\t{test_mape * 100:.2f}%"
    )

    log.info("Performance for each prediction step:")
    test_pred_tensor = tf.convert_to_tensor(test_pred)
    testY_tensor = tf.convert_to_tensor(testY)

    for q in range(Q):
        mae, rmse, mape = metric(test_pred_tensor[:, q], testY_tensor[:, q])
        log.info(
            f"  - Step {q + 1:02d}:    {mae:.2f}\t\t{rmse:.2f}\t\t{mape * 100:.2f}%"
        )


def main():
    """主函数"""
    log = setup_environment(args.log_file, args.use_mixed_precision)

    # Load data
    log.info("**** Loading data ****")
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

    log.info(f"trainX: {trainX.shape}, trainY: {trainY.shape}")
    log.info(f"valX: {valX.shape}, valY: {valY.shape}")
    log.info(f"testX: {testX.shape}, testY: {testY.shape}")

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
        args.use_mixed_precision,
        log,
    )

    # Build model
    log.info("**** Building GMAN model (QUICK) ****")
    model = GMAN(args, SE, mean, std, bn=False)
    
    # Get steps per epoch (before summary to avoid build issues)
    steps_per_epoch = len(train_ds)
    
    # Explicitly build model before summary
    compute_dtype = tf.float16 if args.use_mixed_precision else tf.float32
    input_shape = (
        (args.batch_size, args.P, trainX.shape[-1]),  # X shape
        (args.batch_size, args.P + args.Q, 2),         # TE shape
    )
    model.build(input_shape)
    
    # Display model summary
    log.info("Model architecture:")
    model.summary()

    # Train model
    history, log_dir = build_and_train_model(
        log=log,
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        max_epoch=args.max_epoch,
        patience=args.patience,
        model_file=args.model_file,
        learning_rate=args.learning_rate,
        decay_epoch=args.decay_epoch,
        steps_per_epoch=steps_per_epoch,
    )

    # Evaluate model
    evaluate_model(
        log=log,
        model=model,
        test_ds=test_ds,
        testY=testY,
        Q=args.Q,
        model_file=args.model_file,
    )

    log.info("=" * 80)
    log.info("QUICK TRAINING COMPLETE!")
    log.info(f"Quick model saved to: {args.model_file}")
    log.info("Next steps:")
    log.info("  1. Review results with: tensorboard --logdir=logs/fit_quick")
    log.info("  2. If satisfied, use this as warm start for full training")
    log.info("  3. Or adjust hyperparameters and retrain")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
