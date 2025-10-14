import os
import argparse
import numpy as np
import tensorflow as tf
import keras
from model import GMAN, MaskedMAELoss
from utils import load_data, log_string, metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_slot', type=int, default=5, help='time interval')
    parser.add_argument('--P', type=int, default=12, help='history steps')
    parser.add_argument('--Q', type=int, default=12, help='prediction steps')
    parser.add_argument('--L', type=int, default=5, help='number of STAtt Blocks')
    parser.add_argument('--K', type=int, default=8, help='number of attention heads')
    parser.add_argument('--d', type=int, default=8, help='dims of each head attention outputs')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='testing set ratio')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--max_epoch', type=int, default=100, help='epoch to run')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stop')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=5, help='decay epoch')
    parser.add_argument('--traffic_file', default='./data/METR-LA/metr-la.h5', help='traffic file')
    parser.add_argument('--SE_file', default='./data/METR-LA/SE(METR).txt', help='spatial embedding file')
    parser.add_argument('--model_file', default='./models/GMAN.weights.h5', help='save the model to disk')
    parser.add_argument('--log_file', default='./log/log', help='log file')
    parser.add_argument('--use_mixed_precision', type=bool, default=True, help='use mixed precision training')
    args = parser.parse_args()

    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('log'):
        os.makedirs('log')

    with open(args.log_file, 'w') as log:
        log_string(log, str(args))
        
        # Configure GPU - M4 GPU is always available
        gpus = tf.config.list_physical_devices('GPU')
        log_string(log, f'Detected {len(gpus)} GPU device(s)')
        for gpu in gpus:
            log_string(log, f'  - {gpu}')
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Enable mixed precision for better performance and stability
        if args.use_mixed_precision:
            keras.mixed_precision.set_global_policy('mixed_float16')
            log_string(log, 'Mixed precision (float16) enabled')
        else:
            log_string(log, 'Using default precision (float32)')
        
        log_string(log, 'loading data...')
        (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
         SE, mean, std) = load_data(args)
        log_string(log, f'trainX: {trainX.shape}\ttrainY: {trainY.shape}')
        log_string(log, f'valX:   {valX.shape}\t\tvalY:   {valY.shape}')
        log_string(log, f'testX:  {testX.shape}\t\ttestY:  {testY.shape}')
        log_string(log, f'SE:     {SE.shape}')
        log_string(log, f'mean: {mean:.2f}, std: {std:.2f}')
        log_string(log, 'data loaded!')

        # Modern TF2 data pipeline with optimization
        train_ds = tf.data.Dataset.from_tensor_slices(((trainX, trainTE), trainY))
        train_ds = train_ds.shuffle(buffer_size=2048).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_tensor_slices(((valX, valTE), valY))
        val_ds = val_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        
        test_ds = tf.data.Dataset.from_tensor_slices(((testX, testTE), testY))
        test_ds = test_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

        log_string(log, 'building model...')
        model = GMAN(args, SE, mean, std, bn=True)

        # Modern learning rate schedule
        num_train = trainX.shape[0]
        steps_per_epoch = num_train // args.batch_size
        decay_steps = args.decay_epoch * steps_per_epoch

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=decay_steps,
            decay_rate=0.7,
            staircase=True
        )

        # Modern optimizer with mixed precision support
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        
        if args.use_mixed_precision:
            # Wrap optimizer for mixed precision
            optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)
            log_string(log, 'Using LossScaleOptimizer for mixed precision')

        # Compile model with modern Keras 3 API
        model.compile(
            optimizer=optimizer,
            loss=MaskedMAELoss(),
            metrics=[
                keras.metrics.MeanAbsoluteError(name='mae'),
                keras.metrics.RootMeanSquaredError(name='rmse'),
                keras.metrics.MeanAbsolutePercentageError(name='mape')
            ]
        )

        # Build model through dummy forward pass
        log_string(log, 'initializing model weights...')
        try:
            dummy_batch = next(iter(train_ds.take(1)))
            _ = model(dummy_batch[0], training=False)
            total_params = model.count_params()
            log_string(log, f'Model built successfully with {total_params:,} trainable parameters')
        except Exception as e:
            log_string(log, f'Warning: Could not build model in advance: {e}')
            log_string(log, 'Model will be built during first training step')

        log_string(log, '**** training model ****')
        
        # Modern callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=args.patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=args.model_file,
                save_weights_only=True,
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Training
        history = model.fit(
            train_ds,
            epochs=args.max_epoch,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1
        )

        log_string(log, '**** testing model ****')
        model.load_weights(args.model_file)
        log_string(log, 'Best model weights restored!')
        log_string(log, 'evaluating on test set...')

        # Test evaluation
        test_pred = model.predict(test_ds, verbose=0)
        test_mae, test_rmse, test_mape = metric(
            test_pred.reshape(-1, args.Q, testY.shape[-1]), 
            testY
        )
        
        log_string(log, '                MAE\t\tRMSE\t\tMAPE')
        log_string(log, f'test             {test_mae:.2f}\t\t{test_rmse:.2f}\t\t{test_mape * 100:.2f}%')

        # Per-step performance
        log_string(log, 'performance in each prediction step')
        test_pred = model.predict(test_ds, verbose=0)
        MAE, RMSE, MAPE = [], [], []
        for q in range(args.Q):
            mae, rmse, mape = metric(test_pred[:, q], testY[:, q])
            MAE.append(mae)
            RMSE.append(rmse)
            MAPE.append(mape)
            log_string(log, f'step: {q + 1:02d}         {mae:.2f}\t\t{rmse:.2f}\t\t{mape * 100:.2f}%')
        
        average_mae = np.mean(MAE)
        average_rmse = np.mean(RMSE)
        average_mape = np.mean(MAPE)
        log_string(
            log, f'average:         {average_mae:.2f}\t\t{average_rmse:.2f}\t\t{average_mape * 100:.2f}%'
        )
        
        log_string(log, 'Training and evaluation completed successfully!')


if __name__ == '__main__':
    main()