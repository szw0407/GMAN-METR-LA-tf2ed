import os
import argparse
import numpy as np
import tensorflow as tf
import keras
from model import GMAN, MaskedMAELoss
from utils import load_data, log_string, metric
from pydantic import BaseModel, Field



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
    max_epoch: int = Field(10, description="epoch to run")
    patience: int = Field(3, description="patience for early stop")
    learning_rate: float = Field(0.001, description="initial learning rate")
    decay_epoch: int = Field(5, description="decay epoch")
    traffic_file: str = Field("./data/METR-LA/metr-la.h5", description="traffic file")
    SE_file: str = Field("./data/METR-LA/SE(METR).txt", description="spatial embedding file")
    model_file: str = Field("./models/GMAN.weights.h5", description="save the model to disk")
    log_file: str = Field("./log/log", description="log file")
    use_mixed_precision: bool = Field(True, description="use mixed precision training")


args = GMANConfig()


def main():

    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("log"):
        os.makedirs("log")

    with open(args.log_file, "w") as log:
        log_string(log, str(args))

        # Configure GPU - M4 GPU is always available
        gpus = tf.config.list_physical_devices("GPU")
        log_string(log, f"Detected {len(gpus)} GPU device(s)")
        for gpu in gpus:
            log_string(log, f"  - {gpu}")
            tf.config.experimental.set_memory_growth(gpu, True)

        # Enable mixed precision for better performance and stability
        if args.use_mixed_precision:
            keras.mixed_precision.set_global_policy("mixed_float16")
            log_string(log, "Mixed precision (float16) enabled")
        else:
            log_string(log, "Using default precision (float32)")

        log_string(log, "loading data...")
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
        log_string(log, f"trainX: {trainX.shape}\ttrainY: {trainY.shape}")
        log_string(log, f"valX:   {valX.shape}\t\tvalY:   {valY.shape}")
        log_string(log, f"testX:  {testX.shape}\t\ttestY:  {testY.shape}")
        log_string(log, f"SE:     {SE.shape}")
        log_string(log, f"mean: {mean:.2f}, std: {std:.2f}")
        log_string(log, "data loaded!")

        # Modern TF2 data pipeline with optimization
        train_ds = tf.data.Dataset.from_tensor_slices(((trainX, trainTE), trainY))
        train_ds = (
            train_ds.shuffle(buffer_size=2048)
            .batch(args.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        val_ds = tf.data.Dataset.from_tensor_slices(((valX, valTE), valY))
        val_ds = val_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

        test_ds = tf.data.Dataset.from_tensor_slices(((testX, testTE), testY))
        test_ds = test_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

        log_string(log, "building model...")
        model = GMAN(args, SE, mean, std, bn=True)

        # Modern learning rate schedule
        num_train = trainX.shape[0]
        steps_per_epoch = num_train // args.batch_size
        decay_steps = args.decay_epoch * steps_per_epoch

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=decay_steps,
            decay_rate=0.7,
            staircase=True,
        )

        # Modern optimizer with mixed precision support
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

        if args.use_mixed_precision:
            # Wrap optimizer for mixed precision
            optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)
            log_string(log, "Using LossScaleOptimizer for mixed precision")

        # Compile model with modern Keras 3 API
        model.compile(
            optimizer=optimizer,
            loss=MaskedMAELoss(),
            metrics=[
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
                keras.metrics.MeanAbsolutePercentageError(name="mape"),
            ],
        )

        # Build model through dummy forward pass
        log_string(log, "initializing model weights...")
        try:
            dummy_batch = next(iter(train_ds.take(1)))
            _ = model(dummy_batch[0], training=False)
            total_params = model.count_params()
            log_string(
                log,
                f"Model built successfully with {total_params:,} trainable parameters",
            )
        except Exception as e:
            log_string(log, f"Warning: Could not build model in advance: {e}")
            log_string(log, "Model will be built during first training step")

        log_string(log, "**** training model ****")

        # Modern callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=args.model_file,
                save_weights_only=True,
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.7, patience=3, min_lr=1e-6, verbose=1
            ),
        ]

        # Training
        history = model.fit(
            train_ds,
            epochs=args.max_epoch,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1,
        )

        log_string(log, "**** testing model ****")
        model.load_weights(args.model_file)
        log_string(log, "Best model weights restored!")
        log_string(log, "evaluating on test set...")

        # Test evaluation
        test_pred = model.predict(test_ds, verbose=0)
        test_mae, test_rmse, test_mape = metric(
            test_pred.reshape(-1, args.Q, testY.shape[-1]), testY
        )

        log_string(log, "                MAE\t\tRMSE\t\tMAPE")
        log_string(
            log,
            f"test             {test_mae:.2f}\t\t{test_rmse:.2f}\t\t{test_mape * 100:.2f}%",
        )

        # Per-step performance
        log_string(log, "performance in each prediction step")
        test_pred = model.predict(test_ds, verbose=0)
        MAE, RMSE, MAPE = [], [], []
        for q in range(args.Q):
            mae, rmse, mape = metric(test_pred[:, q], testY[:, q])
            MAE.append(mae)
            RMSE.append(rmse)
            MAPE.append(mape)
            log_string(
                log,
                f"step: {q + 1:02d}         {mae:.2f}\t\t{rmse:.2f}\t\t{mape * 100:.2f}%",
            )

        average_mae = np.mean(MAE)
        average_rmse = np.mean(RMSE)
        average_mape = np.mean(MAPE)
        log_string(
            log,
            f"average:         {average_mae:.2f}\t\t{average_rmse:.2f}\t\t{average_mape * 100:.2f}%",
        )

        log_string(log, "Training and evaluation completed successfully!")


if __name__ == "__main__":
    main()