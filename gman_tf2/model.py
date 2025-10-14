import tensorflow as tf
from keras.layers import Layer, Dense, Dropout
import keras
class FC(Layer):
    def __init__(self, units, activations, use_bias=True, drop=None):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            activations = list(activations)
        assert type(units) == list
        self.conv_layers = []
        self.conv_layers.extend(
            keras.layers.Conv2D(
                filters=num_unit,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding='VALID',
                use_bias=use_bias,
                activation=activation,
            )
            for num_unit, activation in zip(units, activations)
        )
        self.drop = drop
        if self.drop is not None:
            self.dropout_layer = Dropout(drop)

    def call(self, x, training=None):
        for conv in self.conv_layers:
            if self.drop is not None:
                x = self.dropout_layer(x, training=training)
            x = conv(x)
        return x

class STEmbedding(Layer):
    def __init__(self, D):
        super(STEmbedding, self).__init__()
        self.D = D
        self.fc_se = FC(units=[D, D], activations=[tf.nn.relu, None])
        self.fc_te = FC(units=[D, D], activations=[tf.nn.relu, None])

    def call(self, SE, TE, T):
        # spatial embedding
        SE = tf.expand_dims(tf.expand_dims(SE, axis=0), axis=0)
        SE = self.fc_se(SE)
        # temporal embedding
        dayofweek = tf.one_hot(TE[..., 0], depth=7)
        timeofday = tf.one_hot(TE[..., 1], depth=T)
        TE = tf.concat((dayofweek, timeofday), axis=-1)
        TE = tf.expand_dims(TE, axis=2)
        TE = self.fc_te(TE)
        return tf.add(SE, TE)

class SpatialAttention(Layer):
    def __init__(self, K, d):
        super(SpatialAttention, self).__init__()
        self.K = K
        self.d = d
        self.D = K * d
        self.query = FC(self.D, tf.nn.relu)
        self.key = FC(self.D, tf.nn.relu)
        self.value = FC(self.D, tf.nn.relu)
        self.fc = FC([self.D, self.D], [tf.nn.relu, None])

    def call(self, X, STE):
        X_ = tf.concat((X, STE), axis=-1)
        query = self.query(X_)
        key = self.key(X_)
        value = self.value(X_)

        query = tf.concat(tf.split(query, self.K, axis=-1), axis=0)
        key = tf.concat(tf.split(key, self.K, axis=-1), axis=0)
        value = tf.concat(tf.split(value, self.K, axis=-1), axis=0)

        attention = tf.matmul(query, key, transpose_b=True)
        attention /= (self.d ** 0.5)
        attention = tf.nn.softmax(attention, axis=-1)

        X = tf.matmul(attention, value)
        X = tf.concat(tf.split(X, self.K, axis=0), axis=-1)
        X = self.fc(X)
        return X

class TemporalAttention(Layer):
    def __init__(self, K, d, mask=True):
        super(TemporalAttention, self).__init__()
        self.K = K
        self.d = d
        self.D = K * d
        self.mask = mask
        self.query = FC(self.D, tf.nn.relu)
        self.key = FC(self.D, tf.nn.relu)
        self.value = FC(self.D, tf.nn.relu)
        self.fc = FC([self.D, self.D], [tf.nn.relu, None])

    def call(self, X, STE):
        X_ = tf.concat((X, STE), axis=-1)
        query = self.query(X_)
        key = self.key(X_)
        value = self.value(X_)

        query = tf.concat(tf.split(query, self.K, axis=-1), axis=0)
        key = tf.concat(tf.split(key, self.K, axis=-1), axis=0)
        value = tf.concat(tf.split(value, self.K, axis=-1), axis=0)

        query = tf.transpose(query, perm=(0, 2, 1, 3))
        key = tf.transpose(key, perm=(0, 2, 3, 1))
        value = tf.transpose(value, perm=(0, 2, 1, 3))

        attention = tf.matmul(query, key)
        attention /= (self.d ** 0.5)

        if self.mask:
            num_step = X.get_shape()[1]
            mask = tf.linalg.LinearOperatorLowerTriangular(tf.ones((num_step, num_step))).to_dense()
            mask = tf.cast(mask, dtype=tf.bool)
            attention = tf.where(mask, attention, -2 ** 15 + 1)

        attention = tf.nn.softmax(attention, axis=-1)

        X = tf.matmul(attention, value)
        X = tf.transpose(X, perm=(0, 2, 1, 3))
        X = tf.concat(tf.split(X, self.K, axis=0), axis=-1)
        X = self.fc(X)
        return X

class GatedFusion(Layer):
    def __init__(self, D):
        super(GatedFusion, self).__init__()
        self.D = D
        self.fc_xs = FC(D, None, use_bias=False)
        self.fc_xt = FC(D, None, use_bias=True)
        self.fc_h = FC([D, D], [tf.nn.relu, None])

    def call(self, HS, HT):
        XS = self.fc_xs(HS)
        XT = self.fc_xt(HT)
        z = tf.nn.sigmoid(tf.add(XS, XT))
        H = tf.add(tf.multiply(z, HS), tf.multiply(1 - z, HT))
        H = self.fc_h(H)
        return H

class STAttBlock(Layer):
    def __init__(self, K, d):
        super(STAttBlock, self).__init__()
        self.spatial_attention = SpatialAttention(K, d)
        self.temporal_attention = TemporalAttention(K, d)
        self.gated_fusion = GatedFusion(K * d)

    def call(self, X, STE):
        HS = self.spatial_attention(X, STE)
        HT = self.temporal_attention(X, STE)
        H = self.gated_fusion(HS, HT)
        return tf.add(X, H)

class TransformAttention(Layer):
    def __init__(self, K, d):
        super(TransformAttention, self).__init__()
        self.K = K
        self.d = d
        self.D = K * d
        self.query = FC(self.D, tf.nn.relu)
        self.key = FC(self.D, tf.nn.relu)
        self.value = FC(self.D, tf.nn.relu)
        self.fc = FC([self.D, self.D], [tf.nn.relu, None])

    def call(self, X, STE_P, STE_Q):
        query = self.query(STE_Q)
        key = self.key(STE_P)
        value = self.value(X)

        query = tf.concat(tf.split(query, self.K, axis=-1), axis=0)
        key = tf.concat(tf.split(key, self.K, axis=-1), axis=0)
        value = tf.concat(tf.split(value, self.K, axis=-1), axis=0)

        query = tf.transpose(query, perm=(0, 2, 1, 3))
        key = tf.transpose(key, perm=(0, 2, 3, 1))
        value = tf.transpose(value, perm=(0, 2, 1, 3))

        attention = tf.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = tf.nn.softmax(attention, axis=-1)

        X = tf.matmul(attention, value)
        X = tf.transpose(X, perm=(0, 2, 1, 3))
        X = tf.concat(tf.split(X, self.K, axis=0), axis=-1)
        X = self.fc(X)
        return X

class GMAN(tf.keras.Model):
    def __init__(self, args, SE, mean, std, **kwargs):
        super(GMAN, self).__init__(**kwargs)
        self.L = args.L
        self.P = args.P
        self.Q = args.Q
        self.T = 24 * 60 // args.time_slot
        D = args.K * args.d
        self.SE = tf.constant(SE, dtype=tf.float32)
        self.mean = mean
        self.std = std
        
        self.fc_x = FC([D, D], [tf.nn.relu, None])
        self.st_embedding = STEmbedding(D)
        self.encoder = [STAttBlock(args.K, args.d) for _ in range(self.L)]
        self.transform_attention = TransformAttention(args.K, args.d)
        self.decoder = [STAttBlock(args.K, args.d) for _ in range(self.L)]
        self.fc_out = FC([D, 1], [tf.nn.relu, None], drop=0.1)

    def call(self, inputs, training=None):
        X, TE = inputs
        X = tf.expand_dims(X, axis=-1)
        X = self.fc_x(X)

        STE = self.st_embedding(self.SE, TE, T=self.T)
        STE_P = STE[:, :self.P]
        STE_Q = STE[:, self.P:]

        for block in self.encoder:
            X = block(X, STE_P)

        X = self.transform_attention(X, STE_P, STE_Q)

        for block in self.decoder:
            X = block(X, STE_Q)

        X = self.fc_out(X, training=training)
        return tf.squeeze(X, axis=3)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            y_pred = y_pred * self.std + self.mean
            loss = self.compute_loss(y=y, y_pred=y_pred)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        y_pred = y_pred * self.std + self.mean
        loss = self.compute_loss(y=y, y_pred=y_pred)
        
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        x = data[0]
        y_pred = self(x, training=False)
        return y_pred * self.std + self.mean
