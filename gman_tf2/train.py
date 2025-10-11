import os
import argparse
import time
import datetime
import numpy as np
import tensorflow as tf
import keras
from model import GMAN
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
    # 加速：增大 batch_size，减少 max_epoch，减少 patience
    parser.add_argument('--batch_size', type=int, default=96, help='batch size (default 128 for speed)')
    parser.add_argument('--max_epoch', type=int, default=100, help='epoch to run (default 30 for speed)')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stop (default 3 for speed)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=5, help='decay epoch')
    parser.add_argument('--traffic_file', default='../data/METR-LA/metr-la.h5', help='traffic file')
    parser.add_argument('--se_file', default='../METR/data/SE(METR).txt', help='spatial embedding file')
    parser.add_argument('--model_file', default='./models/GMAN.weights.h5', help='save the model to disk')
    parser.add_argument('--log_file', default='./log/log', help='log file')
    args = parser.parse_args()

    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('log'):
        os.makedirs('log')

    with open(args.log_file, 'w') as log:
        log_string(log, str(args))
        log_string(log, 'loading data...')
        (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
         se, mean, std) = load_data(args)
        log_string(log, f'trainX: {trainX.shape}\ttrainY: {trainY.shape}')
        log_string(log, f'valX:   {valX.shape}\t\tvalY:   {valY.shape}')
        log_string(log, f'testX:  {testX.shape}\t\ttestY:  {testY.shape}')
        log_string(log, 'data loaded!')

        # --- tf.data pipeline ---
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = tf.data.Dataset.from_tensor_slices(((trainX, trainTE), trainY))
        train_ds = train_ds.cache().shuffle(buffer_size=2048).batch(args.batch_size).prefetch(AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices(((valX, valTE), valY))
        val_ds = val_ds.cache().batch(args.batch_size).prefetch(AUTOTUNE)
        test_ds = tf.data.Dataset.from_tensor_slices(((testX, testTE), testY))
        test_ds = test_ds.batch(args.batch_size).prefetch(AUTOTUNE)

        # --- 混合精度训练 ---
        keras.mixed_precision.set_global_policy('mixed_float16')

        log_string(log, 'compiling model...')
        model = GMAN(args, se, mean, std)
        optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
        model.compile(optimizer=optimizer, loss='mae', metrics=['mae', keras.metrics.RootMeanSquaredError(name='rmse'), 'mape'])

        log_string(log, '**** training model ****')
        # 关闭 tensorboard callback 进一步加速
        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            restore_best_weights=True
        )
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=args.model_file,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        )

        history = model.fit(
            train_ds,
            epochs=args.max_epoch,
            validation_data=val_ds,
            callbacks=[early_stopping_callback, model_checkpoint_callback]
        )

        log_string(log, '**** testing model ****')
        model.load_weights(args.model_file)
        log_string(log, 'model restored!')
        log_string(log, 'evaluating...')

        test_metrics = model.evaluate(test_ds)
        log_string(log, '                MAE\t\tRMSE\t\tMAPE')
        log_string(log, f'test             {test_metrics[1]:.2f}\t\t{test_metrics[2]:.2f}\t\t{test_metrics[3] * 100:.2f}%')

        log_string(log, 'performance in each prediction step')
        test_pred = model.predict(test_ds)
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


if __name__ == '__main__':
    main()