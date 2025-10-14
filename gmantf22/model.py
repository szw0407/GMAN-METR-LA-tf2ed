import tensorflow as tf
import keras
from keras import layers, ops


class FC(layers.Layer):
    """
    Fully Connected layer with modern Keras 3 API.
    Supports mixed precision training with proper normalization.
    """
    def __init__(self, units, activations, use_bias=True, drop=None, bn=False, **kwargs):
        super(FC, self).__init__(**kwargs)
        if isinstance(units, int):
            units = [units]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            activations = list(activations)
        
        self.bn = bn
        self.drop = drop
        self.conv_layers = []
        self.bn_layers = []
        
        for num_unit, activation in zip(units, activations):
            # Use modern Keras 3 Conv2D - dtype will be inherited from mixed precision policy
            self.conv_layers.append(
                layers.Conv2D(
                    filters=num_unit,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding='valid',
                    use_bias=use_bias,
                    activation=None,
                    kernel_initializer='glorot_uniform'
                    # Don't specify dtype - let mixed precision policy handle it
                )
            )
            
            # BatchNormalization for stability
            if activation is not None and bn:
                self.bn_layers.append(
                    layers.BatchNormalization(
                        momentum=0.9,
                        epsilon=1e-5
                        # Don't specify dtype - let mixed precision policy handle it
                    )
                )
            else:
                self.bn_layers.append(None)
                
        self.activations = activations
        
        if self.drop is not None:
            self.dropout_layer = layers.Dropout(drop)

    def call(self, x, training=None):
        for conv, bn_layer, activation in zip(self.conv_layers, self.bn_layers, self.activations):
            if self.drop is not None:
                x = self.dropout_layer(x, training=training)
            
            x = conv(x)
            
            if activation is not None:
                if bn_layer is not None:
                    x = bn_layer(x, training=training)
                x = activation(x)
        
        return x


class MaskedMAELoss(keras.losses.Loss):
    """
    Masked Mean Absolute Error loss with proper handling for mixed precision.
    """
    def __init__(self, **kwargs):
        super(MaskedMAELoss, self).__init__(**kwargs)
    
    def call(self, y_true, y_pred):
        # Convert to float32 for loss computation (critical for mixed precision)
        y_true = ops.cast(y_true, 'float32')
        y_pred = ops.cast(y_pred, 'float32')
        
        # Create mask for non-zero values
        mask = ops.not_equal(y_true, 0.0)
        mask = ops.cast(mask, 'float32')
        
        # Avoid division by zero
        mask_sum = ops.sum(mask)
        mask_sum = ops.maximum(mask_sum, 1.0)
        
        # Compute masked MAE
        mae = ops.abs(y_pred - y_true)
        masked_mae = mae * mask
        
        # Safe mean computation
        loss = ops.sum(masked_mae) / mask_sum
        
        # Ensure finite values
        loss = ops.where(ops.isfinite(loss), loss, 0.0)
        
        return loss


class STEmbedding(layers.Layer):
    """Spatial-Temporal Embedding with normalization support"""
    def __init__(self, D, bn=False, **kwargs):
        super(STEmbedding, self).__init__(**kwargs)
        self.D = D
        self.fc_se = FC(units=[D, D], activations=[tf.nn.relu, None], bn=bn)
        self.fc_te = FC(units=[D, D], activations=[tf.nn.relu, None], bn=bn)

    def call(self, SE, TE, T, training=None):
        # Process spatial embedding
        SE = tf.expand_dims(tf.expand_dims(SE, axis=0), axis=0)
        SE = self.fc_se(SE, training=training)
        
        # Process temporal embedding
        # CRITICAL: Cast to int32 explicitly to prevent overflow in one_hot
        day_indices = ops.cast(TE[..., 0], 'int32')
        time_indices = ops.cast(TE[..., 1], 'int32')
        
        dayofweek = tf.one_hot(day_indices, depth=7, dtype=tf.float32)
        timeofday = tf.one_hot(time_indices, depth=T, dtype=tf.float32)
        TE = tf.concat((dayofweek, timeofday), axis=-1)
        TE = tf.expand_dims(TE, axis=2)
        TE = self.fc_te(TE, training=training)
        
        return tf.add(SE, TE)


class SpatialAttention(layers.Layer):
    """Multi-head Spatial Attention with modern Keras 3"""
    def __init__(self, K, d, bn=False, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.K = K
        self.d = d
        self.D = K * d
        self.query = FC(self.D, tf.nn.relu, bn=bn)
        self.key = FC(self.D, tf.nn.relu, bn=bn)
        self.value = FC(self.D, tf.nn.relu, bn=bn)
        self.fc = FC([self.D, self.D], [tf.nn.relu, None], bn=bn)

    def call(self, X, STE, training=None):
        X_ = tf.concat((X, STE), axis=-1)
        query = self.query(X_, training=training)
        key = self.key(X_, training=training)
        value = self.value(X_, training=training)

        # Multi-head split
        query = tf.concat(tf.split(query, self.K, axis=-1), axis=0)
        key = tf.concat(tf.split(key, self.K, axis=-1), axis=0)
        value = tf.concat(tf.split(value, self.K, axis=-1), axis=0)

        # Scaled dot-product attention
        attention = tf.matmul(query, key, transpose_b=True)
        attention = attention / tf.sqrt(tf.cast(self.d, attention.dtype))
        attention = tf.nn.softmax(attention, axis=-1)

        # Apply attention
        X = tf.matmul(attention, value)
        X = tf.concat(tf.split(X, self.K, axis=0), axis=-1)
        X = self.fc(X, training=training)
        
        return X


class TemporalAttention(layers.Layer):
    """Multi-head Temporal Attention with optional masking"""
    def __init__(self, K, d, bn=False, mask=True, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.K = K
        self.d = d
        self.D = K * d
        self.mask = mask
        self.query = FC(self.D, tf.nn.relu, bn=bn)
        self.key = FC(self.D, tf.nn.relu, bn=bn)
        self.value = FC(self.D, tf.nn.relu, bn=bn)
        self.fc = FC([self.D, self.D], [tf.nn.relu, None], bn=bn)

    def call(self, X, STE, training=None):
        X_ = tf.concat((X, STE), axis=-1)
        query = self.query(X_, training=training)
        key = self.key(X_, training=training)
        value = self.value(X_, training=training)

        # Multi-head split
        query = tf.concat(tf.split(query, self.K, axis=-1), axis=0)
        key = tf.concat(tf.split(key, self.K, axis=-1), axis=0)
        value = tf.concat(tf.split(value, self.K, axis=-1), axis=0)

        # Transpose for temporal attention
        query = tf.transpose(query, perm=(0, 2, 1, 3))
        key = tf.transpose(key, perm=(0, 2, 3, 1))
        value = tf.transpose(value, perm=(0, 2, 1, 3))

        # Scaled dot-product attention
        attention = tf.matmul(query, key)
        attention = attention / tf.sqrt(tf.cast(self.d, attention.dtype))

        # Apply causal mask if needed
        if self.mask:
            num_step = ops.shape(query)[2]
            mask_matrix = tf.linalg.band_part(tf.ones((num_step, num_step), dtype=attention.dtype), -1, 0)
            mask_bool = tf.cast(mask_matrix, dtype=tf.bool)
            # CRITICAL: Use -65504.0 instead of -1e9 to prevent overflow in float16
            # Float16 max is ±65504, so -1e9 causes overflow warning
            neg_inf = tf.constant(-65504.0, dtype=attention.dtype)
            attention = tf.where(mask_bool, attention, neg_inf)

        attention = tf.nn.softmax(attention, axis=-1)

        # Apply attention
        X = tf.matmul(attention, value)
        X = tf.transpose(X, perm=(0, 2, 1, 3))
        X = tf.concat(tf.split(X, self.K, axis=0), axis=-1)
        X = self.fc(X, training=training)
        
        return X


class GatedFusion(layers.Layer):
    """Gated fusion mechanism for combining spatial and temporal features"""
    def __init__(self, D, bn=False, **kwargs):
        super(GatedFusion, self).__init__(**kwargs)
        self.D = D
        self.fc_xs = FC(D, None, use_bias=False, bn=bn)
        self.fc_xt = FC(D, None, use_bias=True, bn=bn)
        self.fc_h = FC([D, D], [tf.nn.relu, None], bn=bn)

    def call(self, HS, HT, training=None):
        XS = self.fc_xs(HS, training=training)
        XT = self.fc_xt(HT, training=training)
        z = tf.nn.sigmoid(tf.add(XS, XT))
        # Use tf ops to ensure dtype consistency in mixed precision
        one = tf.ones_like(z)
        H = tf.add(tf.multiply(z, HS), tf.multiply(tf.subtract(one, z), HT))
        H = self.fc_h(H, training=training)
        return H


class STAttBlock(layers.Layer):
    """Spatial-Temporal Attention Block"""
    def __init__(self, K, d, bn=False, **kwargs):
        super(STAttBlock, self).__init__(**kwargs)
        self.spatial_attention = SpatialAttention(K, d, bn=bn)
        self.temporal_attention = TemporalAttention(K, d, bn=bn)
        self.gated_fusion = GatedFusion(K * d, bn=bn)

    def call(self, X, STE, training=None):
        HS = self.spatial_attention(X, STE, training=training)
        HT = self.temporal_attention(X, STE, training=training)
        H = self.gated_fusion(HS, HT, training=training)
        return tf.add(X, H)


class TransformAttention(layers.Layer):
    """Transform attention for encoder-decoder connection"""
    def __init__(self, K, d, bn=False, **kwargs):
        super(TransformAttention, self).__init__(**kwargs)
        self.K = K
        self.d = d
        self.D = K * d
        self.query = FC(self.D, tf.nn.relu, bn=bn)
        self.key = FC(self.D, tf.nn.relu, bn=bn)
        self.value = FC(self.D, tf.nn.relu, bn=bn)
        self.fc = FC([self.D, self.D], [tf.nn.relu, None], bn=bn)

    def call(self, X, STE_P, STE_Q, training=None):
        query = self.query(STE_Q, training=training)
        key = self.key(STE_P, training=training)
        value = self.value(X, training=training)

        # Multi-head split
        query = tf.concat(tf.split(query, self.K, axis=-1), axis=0)
        key = tf.concat(tf.split(key, self.K, axis=-1), axis=0)
        value = tf.concat(tf.split(value, self.K, axis=-1), axis=0)

        # Transpose for attention
        query = tf.transpose(query, perm=(0, 2, 1, 3))
        key = tf.transpose(key, perm=(0, 2, 3, 1))
        value = tf.transpose(value, perm=(0, 2, 1, 3))

        # Scaled dot-product attention
        attention = tf.matmul(query, key)
        attention = attention / tf.sqrt(tf.cast(self.d, attention.dtype))
        attention = tf.nn.softmax(attention, axis=-1)

        # Apply attention
        X = tf.matmul(attention, value)
        X = tf.transpose(X, perm=(0, 2, 1, 3))
        X = tf.concat(tf.split(X, self.K, axis=0), axis=-1)
        X = self.fc(X, training=training)
        
        return X


class GMAN(keras.Model):
    """
    Graph Multi-Attention Network (GMAN) for traffic prediction.
    Modern TensorFlow 2 / Keras 3 implementation with mixed precision support.
    """
    def __init__(self, args, SE, mean, std, bn=False, **kwargs):
        super(GMAN, self).__init__(**kwargs)
        self.L = args.L
        self.P = args.P
        self.Q = args.Q
        self.T = 24 * 60 // args.time_slot
        D = args.K * args.d
        
        # Store normalization parameters as constants (float32)
        # CRITICAL: Convert to Python float first to prevent overflow in tf.constant
        self.SE = tf.constant(SE, dtype=tf.float32)
        self.mean = tf.constant(float(mean), dtype=tf.float32)
        self.std = tf.constant(float(std), dtype=tf.float32)
        
        # Build model layers
        self.fc_x = FC([D, D], [tf.nn.relu, None], bn=bn)
        self.st_embedding = STEmbedding(D, bn=bn)
        self.encoder = [STAttBlock(args.K, args.d, bn=bn) for _ in range(self.L)]
        self.transform_attention = TransformAttention(args.K, args.d, bn=bn)
        self.decoder = [STAttBlock(args.K, args.d, bn=bn) for _ in range(self.L)]
        self.fc_out = FC([D, 1], [tf.nn.relu, None], drop=0.1, bn=bn)

    def call(self, inputs, training=None):
        """Forward pass through the network"""
        X, TE = inputs
        
        # Expand dimensions and apply initial transformation
        X = tf.expand_dims(X, axis=-1)
        X = self.fc_x(X, training=training)

        # Generate spatial-temporal embeddings
        STE = self.st_embedding(self.SE, TE, T=self.T, training=training)
        STE_P = STE[:, :self.P]
        STE_Q = STE[:, self.P:]

        # Encoder: process historical data
        for block in self.encoder:
            X = block(X, STE_P, training=training)

        # Transform attention: connect encoder and decoder
        X = self.transform_attention(X, STE_P, STE_Q, training=training)

        # Decoder: generate predictions
        for block in self.decoder:
            X = block(X, STE_Q, training=training)

        # Output layer
        X = self.fc_out(X, training=training)
        return tf.squeeze(X, axis=3)

    def train_step(self, data):
        """Custom training step with mixed precision support"""
        x, y = data
        
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)
            
            # Denormalize predictions to original scale (float32)
            y_pred = ops.cast(y_pred, 'float32') * self.std + self.mean
            y = ops.cast(y, 'float32')
            
            # Compute loss
            loss = self.compute_loss(y=y, y_pred=y_pred)
            
            # Handle mixed precision scaling
            if hasattr(self.optimizer, 'get_scaled_loss'):
                scaled_loss = self.optimizer.get_scaled_loss(loss)
            else:
                scaled_loss = loss
        
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(scaled_loss, trainable_vars)
        
        # Unscale gradients if using mixed precision
        if hasattr(self.optimizer, 'get_unscaled_gradients'):
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        
        # Clip gradients for stability
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        """Custom test step with proper denormalization"""
        x, y = data
        
        # Forward pass
        y_pred = self(x, training=False)
        
        # Denormalize to original scale
        y_pred = ops.cast(y_pred, 'float32') * self.std + self.mean
        y = ops.cast(y, 'float32')
        
        # Compute loss
        loss = self.compute_loss(y=y, y_pred=y_pred)
        
        # Update metrics
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}
    
    def predict_step(self, data):
        """Custom predict step with denormalization"""
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data
            
        # Forward pass
        y_pred = self(x, training=False)
        
        # Denormalize to original scale
        y_pred = ops.cast(y_pred, 'float32') * self.std + self.mean
        
        return y_pred