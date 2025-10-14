import logging
from pathlib import Path


import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime

from model import GMAN, MaskedMAELoss
from utils import GMANConfig, load_data, metric

# Initialize configuration from the utility class
args = GMANConfig()  # type: ignore


def setup_environment(log_file: str, use_mixed_precision: bool):
    """
    Configures the runtime environment: logging, GPU, and mixed precision.
    """
    # Setup logging
    Path("log").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    logging.info("Logging setup complete.")
    logging.info(str(args))

    # Configure GPU devices
    gpus = tf.config.list_physical_devices("GPU")
    logging.info(f"Detected {len(gpus)} GPU device(s).")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f"  - {gpu.name}: Memory growth enabled.")

    # Configure mixed precision
    if use_mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
        logging.info("Mixed precision (float16) enabled.")
    else:
        logging.info("Using default precision (float32).")


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
):
    """
    Creates tf.data.Dataset pipelines for training, validation, and testing.
    """

    def create_dataset(X, TE, Y, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices(((X, TE), Y))
        if shuffle:
            ds = ds.shuffle(buffer_size=2048)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()

    train_ds = create_dataset(trainX, trainTE, trainY, shuffle=True)
    val_ds = create_dataset(valX, valTE, valY)
    test_ds = create_dataset(testX, testTE, testY)

    logging.info("tf.data.Dataset pipelines created.")
    return train_ds, val_ds, test_ds


def build_and_train_model(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    max_epoch: int,
    patience: int,
    model_file: str,
):
    """
    Compiles, trains, and saves the GMAN model.
    """
    logging.info("**** Training model ****")

    # Modern callbacks

    # TensorBoard log dir
    log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
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
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.7, patience=3, min_lr=1e-6, verbose=1
        ),
        tb_callback,
    ]

    # Training

    history = model.fit(
        train_ds,
        epochs=max_epoch,
        validation_data=val_ds,
        callbacks=callbacks,
    )
    logging.info("Model training complete.")

    # 可视化训练过程
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(history.history['loss'], label='Train Loss')
    ax[0].plot(history.history['val_loss'], label='Val Loss')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    if 'mae' in history.history:
        ax[1].plot(history.history['mae'], label='Train MAE')
        ax[1].plot(history.history['val_mae'], label='Val MAE')
    if 'mape' in history.history:
        ax[1].plot(history.history['mape'], label='Train MAPE')
        ax[1].plot(history.history['val_mape'], label='Val MAPE')
    if 'rmse' in history.history:
        ax[1].plot(history.history['rmse'], label='Train RMSE')
        ax[1].plot(history.history['val_rmse'], label='Val RMSE')
    ax[1].set_title('Metrics')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Value')
    ax[1].legend()
    plt.tight_layout()
    fig_path = f"{log_dir}/training_curves.png"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path)
    plt.close(fig)
    logging.info(f"Training curves saved to {fig_path}")
    return history


def evaluate_model(
    model: keras.Model,
    test_ds: tf.data.Dataset,
    testY: np.ndarray,
    Q: int,
    model_file: str,
):
    """
    Evaluates the model on the test set and logs the performance.
    """
    logging.info("**** Testing model ****")
    model.load_weights(model_file)
    logging.info("Best model weights restored for testing.")

    # Perform prediction once
    test_pred = model.predict(test_ds)

    # Overall performance
    test_mae, test_rmse, test_mape = metric(test_pred, testY)
    logging.info("                MAE\t\tRMSE\t\tMAPE")
    logging.info(
        f"Overall Test     {test_mae:.2f}\t\t{test_rmse:.2f}\t\t{test_mape * 100:.2f}%"
    )

    # Per-step performance
    logging.info("Performance for each prediction step:")
    mae_steps, rmse_steps, mape_steps = [], [], []
    for q in range(Q):
        mae, rmse, mape = metric(test_pred[:, q], testY[:, q])
        mae_steps.append(mae)
        rmse_steps.append(rmse)
        mape_steps.append(mape)
        logging.info(
            f"  - Step {q + 1:02d}:    {mae:.2f}\t\t{rmse:.2f}\t\t{mape * 100:.2f}%"
        )

    # Average per-step performance
    avg_mae = np.mean(mae_steps)
    avg_rmse = np.mean(rmse_steps)
    avg_mape = np.mean(mape_steps)
    logging.info(
        f"Average Steps:   {avg_mae:.2f}\t\t{avg_rmse:.2f}\t\t{avg_mape * 100:.2f}%"
    )


def main():
    """
    Main function to run the GMAN model training and evaluation pipeline.
    """
    Path("models").mkdir(exist_ok=True)
    setup_environment(args.log_file, args.use_mixed_precision)

    logging.info("Loading data...")
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
    logging.info(
        f"Data loaded. Shapes:\n"
        f"  trainX: {trainX.shape}, trainY: {trainY.shape}\n"
        f"  valX:   {valX.shape}, valY:   {valY.shape}\n"
        f"  testX:  {testX.shape}, testY:  {testY.shape}\n"
        f"  SE:     {SE.shape}\n"
        f"  Mean: {mean:.2f}, Std: {std:.2f}"
    )

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
    )

    logging.info("Building model...")
    model = GMAN(args, SE, mean, std, bn=True)

    steps_per_epoch = len(trainX) // args.batch_size
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=args.decay_epoch * steps_per_epoch,
        decay_rate=0.7,
        staircase=True,
    )

    # Compile model with modern Keras 3 API
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, global_clipnorm=5.0),  # type: ignore
        loss=MaskedMAELoss(),
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
            keras.metrics.MeanAbsolutePercentageError(name="mape"),
        ],
    )

    build_and_train_model(
        model, train_ds, val_ds, args.max_epoch, args.patience, args.model_file
    )
    # log the model summary
    model.summary(print_fn=lambda x: logging.info(x))
    logging.info("Model built and compiled.")
    evaluate_model(model, test_ds, testY, args.Q, args.model_file)

    logging.info("Training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
