import os
import argparse
import time
import datetime
import math
import numpy as np
import tensorflow as tf
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
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--max_epoch', type=int, default=1000, help='epoch to run')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stop')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=5, help='decay epoch')
    parser.add_argument('--traffic_file', default='../data/METR-LA/metr-la.h5', help='traffic file')
    parser.add_argument('--se_file', default='../METR/data/SE(METR).txt', help='spatial embedding file')
    parser.add_argument('--model_file', default='./models/GMAN', help='save the model to disk')
    parser.add_argument('--log_file', default='./log/log', help='log file')
    args = parser.parse_args()

    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('log'):
        os.makedirs('log')

    log = open(args.log_file, 'w')
    log_string(log, str(args))

    log_string(log, 'loading data...')
    (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
     se, mean, std) = load_data(args)
    log_string(log, f'trainX: {trainX.shape}\ttrainY: {trainY.shape}')
    log_string(log, f'valX:   {valX.shape}\t\tvalY:   {valY.shape}')
    log_string(log, f'testX:  {testX.shape}\t\ttestY:  {testY.shape}')
    log_string(log, 'data loaded!')

    log_string(log, 'compiling model...')
    model = GMAN(args, se)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    def loss_fn(pred, label):
        return tf.reduce_mean(tf.abs(pred - label))

    @tf.function
    def train_step(x, te, y):
        with tf.GradientTape() as tape:
            pred = model((x, te), training=True)
            pred = pred * std + mean
            loss = loss_fn(pred, y)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @tf.function
    def eval_step(x, te):
        pred = model((x, te), training=False)
        return pred * std + mean

    log_string(log, '**** training model ****')
    num_train = trainX.shape[0]
    num_val = valX.shape[0]
    wait = 0
    val_loss_min = np.inf

    for epoch in range(args.max_epoch):
        if wait >= args.patience:
            log_string(log, f'early stop at epoch: {epoch:04d}')
            break

        # Shuffle training data
        permutation = np.random.permutation(num_train)
        trainX = trainX[permutation]
        trainTE = trainTE[permutation]
        trainY = trainY[permutation]

        # Train
        start_train = time.time()
        train_loss = 0
        num_batch = math.ceil(num_train / args.batch_size)
        for batch_idx in range(num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
            loss = train_step(
                tf.constant(trainX[start_idx:end_idx], dtype=tf.float32),
                tf.constant(trainTE[start_idx:end_idx], dtype=tf.int32),
                tf.constant(trainY[start_idx:end_idx], dtype=tf.float32)
            )
            train_loss += loss.numpy() * (end_idx - start_idx)
        train_loss /= num_train
        end_train = time.time()

        # Validation
        start_val = time.time()
        val_loss = 0
        num_batch = math.ceil(num_val / args.batch_size)
        val_pred_all = []
        for batch_idx in range(num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
            pred = eval_step(
                tf.constant(valX[start_idx:end_idx], dtype=tf.float32),
                tf.constant(valTE[start_idx:end_idx], dtype=tf.int32)
            )
            val_pred_all.append(pred.numpy())
        
        val_pred_all = np.concatenate(val_pred_all, axis=0)
        val_loss = loss_fn(val_pred_all, valY).numpy()
        end_val = time.time()

        log_string(
            log,
            f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
            f'epoch: {epoch + 1:04d}/{args.max_epoch}, '
            f'training time: {end_train - start_train:.1f}s, '
            f'inference time: {end_val - start_val:.1f}s'
        )
        log_string(log, f'train loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')

        if val_loss <= val_loss_min:
            log_string(
                log,
                f'val loss decrease from {val_loss_min:.4f} to {val_loss:.4f}, '
                f'saving model to {args.model_file}'
            )
            wait = 0
            val_loss_min = val_loss
            model.save_weights(args.model_file)
        else:
            wait += 1

    log_string(log, '**** testing model ****')
    model.load_weights(args.model_file)
    log_string(log, 'model restored!')
    log_string(log, 'evaluating...')

    num_test = testX.shape[0]
    test_pred_all = []
    num_batch = math.ceil(num_test / args.batch_size)
    start_test = time.time()
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
        pred = eval_step(
            tf.constant(testX[start_idx:end_idx], dtype=tf.float32),
            tf.constant(testTE[start_idx:end_idx], dtype=tf.int32)
        )
        test_pred_all.append(pred.numpy())
    end_test = time.time()
    test_pred_all = np.concatenate(test_pred_all, axis=0)

    test_mae, test_rmse, test_mape = metric(test_pred_all, testY)
    log_string(log, f'testing time: {end_test - start_test:.1f}s')
    log_string(log, '                MAE\t\tRMSE\t\tMAPE')
    log_string(log, f'test             {test_mae:.2f}\t\t{test_rmse:.2f}\t\t{test_mape * 100:.2f}%')

    log_string(log, 'performance in each prediction step')
    MAE, RMSE, MAPE = [], [], []
    for q in range(args.Q):
        mae, rmse, mape = metric(test_pred_all[:, q], testY[:, q])
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

    log.close()

if __name__ == '__main__':
    main()
