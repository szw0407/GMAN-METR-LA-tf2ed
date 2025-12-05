import tensorflow as tf
import keras
from keras import layers, ops


@keras.saving.register_keras_serializable()
class FC(layers.Layer):
    """
    使用 Conv2D 实现的高效全连接层，支持多层堆叠。

    该实现利用 1x1 卷积，可以在保持空间或时间维度的同时，对特征维度进行变换，
    这比在每个位置上独立应用 Dense 层要高效得多。
    """

    def __init__(
        self, units, activations, use_bias=True, drop=None, bn=False, **kwargs
    ):
        super().__init__(**kwargs)
        if isinstance(units, int):
            units = [units]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            activations = list(activations)

        self.bn = bn
        self.drop_rate = drop
        self.conv_layers = []
        self.bn_layers = []

        for num_unit in units:
            # 使用 Keras 3 的 Conv2D
            self.conv_layers.append(
                layers.Conv2D(
                    filters=num_unit,
                    kernel_size=(1, 1),
                    padding="valid",
                    use_bias=use_bias,
                    kernel_initializer="glorot_uniform",
                    activation=None,  # 手动应用激活，以便插入BN
                )
            )

            # 如果需要，添加 BatchNormalization 层
            if self.bn:
                self.bn_layers.append(layers.BatchNormalization())
            else:
                self.bn_layers.append(None)

        self.activations = [keras.activations.get(act) for act in activations]

        if self.drop_rate is not None:
            self.dropout_layer = layers.Dropout(self.drop_rate)

    def call(self, x, training=None):
        for conv, bn_layer, activation in zip(
            self.conv_layers, self.bn_layers, self.activations
        ):

            x = conv(x)

            if bn_layer is not None:
                x = bn_layer(x, training=training)

            if activation is not None:
                x = activation(x)

            if self.drop_rate is not None:
                x = self.dropout_layer(x, training=training)

        return x

    def get_config(self):
        config = super().get_config()
        # 将构造函数参数保存到配置中，使用 Keras 3 序列化方式
        config.update(
            {
                "units": [layer.filters for layer in self.conv_layers],
                "activations": [
                    keras.activations.serialize(act) if act is not None else None 
                    for act in self.activations
                ],
                "use_bias": any(layer.use_bias for layer in self.conv_layers),
                "drop": self.drop_rate,
                "bn": self.bn,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class MaskedMAELoss(keras.losses.Loss):
    """
    带掩码的平均绝对误差损失函数。

    此损失函数会忽略真实标签 `y_true` 中值为 0 的元素，这在处理稀疏数据
    或填充值时非常有用。代码已经对混合精度做了很好的处理。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        # 为保证精度，损失计算强制使用 float32
        y_true_f32 = ops.cast(y_true, "float32")
        y_pred_f32 = ops.cast(y_pred, "float32")
        mask = ops.not_equal(y_true_f32, 0.0)
        mask = ops.cast(mask, "float32")
        
        # 改进掩码归一化，避免除以过小值
        mask_mean = tf.reduce_mean(mask)
        mask_mean = tf.maximum(mask_mean, 1e-5)  # 防止除零
        mask = ops.divide(mask, mask_mean)
        
        # 计算掩码MAE
        masked_error = ops.multiply(ops.abs(y_pred_f32 - y_true_f32), mask)
        mask_sum = tf.maximum(ops.sum(mask), 1.0)  # 防止除零
        loss = ops.divide(ops.sum(masked_error), mask_sum)
        
        # 裁剪损失到合理范围，防止梯度爆炸
        loss = tf.clip_by_value(loss, 0.0, 1000.0)

        # 确保损失值是有限的
        return ops.where(ops.isfinite(loss), loss, 0.0)

    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable()
class MaskedMAE(keras.metrics.Metric):
    """带掩码的MAE指标，与MaskedMAELoss使用相同的掩码逻辑"""
    
    def __init__(self, name="masked_mae", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros", dtype="float32")
        self.count = self.add_weight(name="count", initializer="zeros", dtype="float32")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")
        
        mask = ops.not_equal(y_true, 0.0)
        mask = ops.cast(mask, "float32")
        
        masked_error = ops.multiply(ops.abs(y_pred - y_true), mask)
        self.total.assign_add(ops.sum(masked_error))
        self.count.assign_add(ops.sum(mask))
    
    def result(self):
        return ops.divide(self.total, tf.maximum(self.count, 1.0))
    
    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


@keras.saving.register_keras_serializable()
class MaskedRMSE(keras.metrics.Metric):
    """带掩码的RMSE指标"""
    
    def __init__(self, name="masked_rmse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros", dtype="float32")
        self.count = self.add_weight(name="count", initializer="zeros", dtype="float32")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")
        
        mask = ops.not_equal(y_true, 0.0)
        mask = ops.cast(mask, "float32")
        
        squared_error = ops.multiply(ops.square(y_pred - y_true), mask)
        self.total.assign_add(ops.sum(squared_error))
        self.count.assign_add(ops.sum(mask))
    
    def result(self):
        mse = ops.divide(self.total, tf.maximum(self.count, 1.0))
        return ops.sqrt(mse)
    
    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


@keras.saving.register_keras_serializable()
class MaskedMAPE(keras.metrics.Metric):
    """带掩码的MAPE指标，与utils.metric()使用相同逻辑"""
    
    def __init__(self, name="masked_mape", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros", dtype="float32")
        self.count = self.add_weight(name="count", initializer="zeros", dtype="float32")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")
        
        mask = ops.not_equal(y_true, 0.0)
        mask = ops.cast(mask, "float32")
        
        # 添加epsilon防止除以接近零的值
        epsilon = 1e-3
        percentage_error = ops.abs((y_pred - y_true) / tf.maximum(y_true, epsilon))
        masked_error = ops.multiply(percentage_error, mask)
        
        self.total.assign_add(ops.sum(masked_error))
        self.count.assign_add(ops.sum(mask))
    
    def result(self):
        mape = ops.divide(self.total, tf.maximum(self.count, 1.0))
        # 裁剪到合理范围防止溢出
        return tf.clip_by_value(mape, 0.0, 2.0)
    
    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


@keras.saving.register_keras_serializable()
class STEmbedding(layers.Layer):
    """时空嵌入层 (Spatial-Temporal Embedding)"""

    def __init__(self, D, bn=False, **kwargs):
        super().__init__(**kwargs)
        self.D = D
        self.bn = bn
        # --- 优化点 2: 使用 Keras 字符串别名 ---
        self.fc_se = FC(units=[D, D], activations=["relu", None], bn=bn)
        self.fc_te = FC(units=[D, D], activations=["relu", None], bn=bn)

    def call(self, SE, TE, T, training=None):
        # 空间嵌入 (Spatial Embedding)
        # SE shape: (N, D_se) -> (1, 1, N, D_se)
        SE = ops.expand_dims(ops.expand_dims(SE, axis=0), axis=0)
        SE = self.fc_se(SE, training=training)  # -> (1, 1, N, D)

        # 时间嵌入 (Temporal Embedding)
        # TE shape: (B, P+Q, 2)
        # 显式转换为 int32 以防止 one_hot 溢出
        day_indices = ops.cast(TE[..., 0], "int32")
        time_indices = ops.cast(TE[..., 1], "int32")

        dayofweek = tf.one_hot(day_indices, depth=7, dtype=self.compute_dtype)
        timeofday = tf.one_hot(time_indices, depth=T, dtype=self.compute_dtype)

        TE = ops.concatenate([dayofweek, timeofday], axis=-1)  # -> (B, P+Q, 7+T)
        TE = ops.expand_dims(TE, axis=2)  # -> (B, P+Q, 1, 7+T)
        TE = self.fc_te(TE, training=training)  # -> (B, P+Q, 1, D)

        # 通过广播将 SE 和 TE 相加
        return SE + TE

    def get_config(self):
        config = super().get_config()
        config.update({"D": self.D, "bn": self.bn})
        return config


@keras.saving.register_keras_serializable()
class SpatialAttention(layers.Layer):
    """多头空间注意力 (Multi-head Spatial Attention)"""

    def __init__(self, K, d, bn=False, **kwargs):
        super().__init__(**kwargs)
        self.K = K  # Number of heads
        self.d = d  # Dimension of each head
        self.D = K * d
        self.bn = bn

        self.query = FC(self.D, "relu", bn=bn)
        self.key = FC(self.D, "relu", bn=bn)
        self.value = FC(self.D, "relu", bn=bn)
        self.fc = FC([self.D, self.D], ["relu", None], bn=bn)

    def call(self, X, STE, training=None):
        # Input X shape: (B, P, N, D), STE shape: (B, P, N, D)
        X_ = ops.concatenate([X, STE], axis=-1)

        query = self.query(X_, training=training)  # -> (B, P, N, D)
        key = self.key(X_, training=training)  # -> (B, P, N, D)
        value = self.value(X_, training=training)  # -> (B, P, N, D)

        # --- 优化点 1: 使用 reshape 和 transpose 实现多头注意力 ---
        # 避免在批次维度上使用 split 和 concat
        B, P, N, _ = ops.shape(X)

        # Reshape and transpose for multi-head attention
        # (B, P, N, D) -> (B, P, N, K, d) -> (B, K, P, N, d)
        query = ops.transpose(
            ops.reshape(query, (B, P, N, self.K, self.d)), (0, 3, 1, 2, 4)
        )
        key = ops.transpose(
            ops.reshape(key, (B, P, N, self.K, self.d)), (0, 3, 1, 2, 4)
        )
        value = ops.transpose(
            ops.reshape(value, (B, P, N, self.K, self.d)), (0, 3, 1, 2, 4)
        )

        # Scaled Dot-Product Attention
        # (B, K, P, N, d) @ (B, K, P, d, N) -> (B, K, P, N, N)
        key_transposed = ops.transpose(key, axes=[0, 1, 2, 4, 3])
        attention_scores = ops.matmul(query, key_transposed)
        attention_scores = ops.divide(
            attention_scores, ops.sqrt(ops.cast(self.d, attention_scores.dtype))
        )

        attention_weights = ops.softmax(attention_scores, axis=-1)

        # Apply attention to value
        # (B, K, P, N, N) @ (B, K, P, N, d) -> (B, K, P, N, d)
        attended_value = ops.matmul(attention_weights, value)

        # Concatenate heads
        # (B, K, P, N, d) -> (B, P, N, K, d) -> (B, P, N, D)
        concatenated_value = ops.reshape(
            ops.transpose(attended_value, (0, 2, 3, 1, 4)), (B, P, N, self.D)
        )

        return self.fc(concatenated_value, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"K": self.K, "d": self.d, "bn": self.bn})
        return config


@keras.saving.register_keras_serializable()
class TemporalAttention(layers.Layer):
    """多头时间注意力 (Multi-head Temporal Attention)"""

    def __init__(self, K, d, bn=False, mask=True, **kwargs):
        super().__init__(**kwargs)
        self.K = K
        self.d = d
        self.D = K * d
        self.mask = mask
        self.bn = bn

        self.query = FC(self.D, "relu", bn=bn)
        self.key = FC(self.D, "relu", bn=bn)
        self.value = FC(self.D, "relu", bn=bn)
        self.fc = FC([self.D, self.D], ["relu", None], bn=bn)

    def call(self, X, STE, training=None):
        # Input X shape: (B, P, N, D), STE shape: (B, P, N, D)
        X_ = ops.concatenate([X, STE], axis=-1)

        query = self.query(X_, training=training)
        key = self.key(X_, training=training)
        value = self.value(X_, training=training)

        B, P, N, _ = ops.shape(X)

        # --- 优化点 1: 使用 reshape 和 transpose 实现多头注意力 ---
        # Reshape and transpose for multi-head attention
        # (B, P, N, D) -> (B, P, N, K, d) -> (B, K, N, P, d)
        query = ops.transpose(
            ops.reshape(query, (B, P, N, self.K, self.d)), (0, 3, 2, 1, 4)
        )
        key = ops.transpose(
            ops.reshape(key, (B, P, N, self.K, self.d)), (0, 3, 2, 1, 4)
        )
        value = ops.transpose(
            ops.reshape(value, (B, P, N, self.K, self.d)), (0, 3, 2, 1, 4)
        )

        # Scaled Dot-Product Attention
        # (B, K, N, P, d) @ (B, K, N, d, P) -> (B, K, N, P, P)
        # attention_scores = ops.matmul(query, key, transpose_b=True)
        attention_scores = ops.matmul(query, ops.transpose(key, axes=[0, 1, 2, 4, 3]))
        attention_scores = ops.divide(
            attention_scores, ops.sqrt(ops.cast(self.d, attention_scores.dtype))
        )

        # Apply causal mask if needed
        if self.mask:
            # `band_part` is efficient for creating a lower-triangular matrix
            mask_matrix = tf.linalg.band_part(
                tf.ones((P, P), dtype=attention_scores.dtype), -1, 0  # type: ignore
            )
            # 使用一个非常大的负数进行掩码，softmax 后会变为 0
            # -65504.0 是 float16 的安全值，防止溢出
            neg_inf = tf.constant(-65504.0, dtype=attention_scores.dtype)
            attention_scores = (
                attention_scores * mask_matrix + (1 - mask_matrix) * neg_inf
            )

        attention_weights = ops.softmax(attention_scores, axis=-1)

        # Apply attention to value
        # (B, K, N, P, P) @ (B, K, N, P, d) -> (B, K, N, P, d)
        attended_value = ops.matmul(attention_weights, value)

        # Concatenate heads
        # (B, K, N, P, d) -> (B, P, N, K, d) -> (B, P, N, D)
        concatenated_value = ops.reshape(
            ops.transpose(attended_value, (0, 3, 2, 1, 4)), (B, P, N, self.D)
        )

        return self.fc(concatenated_value, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"K": self.K, "d": self.d, "bn": self.bn, "mask": self.mask})
        return config


@keras.saving.register_keras_serializable()
class GatedFusion(layers.Layer):
    """门控融合机制 (Gated Fusion)"""

    def __init__(self, D, bn=False, **kwargs):
        super().__init__(**kwargs)
        self.D = D
        self.bn = bn
        self.fc_xs = FC(D, None, use_bias=False, bn=bn)
        self.fc_xt = FC(D, None, use_bias=True, bn=bn)
        self.fc_h = FC([D, D], ["relu", None], bn=bn)

    def call(self, HS, HT, training=None):
        # HS: spatial attention output, HT: temporal attention output
        XS = self.fc_xs(HS, training=training)
        XT = self.fc_xt(HT, training=training)

        z = keras.activations.sigmoid(XS + XT)
        # H = z * HS + (1 - z) * HT
        H = ops.add(ops.multiply(z, HS), ops.multiply(ops.subtract(1.0, z), HT))
        return self.fc_h(H, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"D": self.D, "bn": self.bn})
        return config


@keras.saving.register_keras_serializable()
class STAttBlock(layers.Layer):
    """时空注意力模块 (Spatial-Temporal Attention Block)"""

    def __init__(self, K, d, bn=False, **kwargs):
        super().__init__(**kwargs)
        self.K = K
        self.d = d
        self.bn = bn
        self.spatial_attention = SpatialAttention(K, d, bn=bn)
        self.temporal_attention = TemporalAttention(K, d, bn=bn)
        self.gated_fusion = GatedFusion(K * d, bn=bn)

    def call(self, X, STE, training=None):
        HS = self.spatial_attention(X, STE, training=training)
        HT = self.temporal_attention(X, STE, training=training)
        H = self.gated_fusion(HS, HT, training=training)
        # Residual connection
        return X + H

    def get_config(self):
        config = super().get_config()
        config.update({"K": self.K, "d": self.d, "bn": self.bn})
        return config


@keras.saving.register_keras_serializable()
class TransformAttention(layers.Layer):
    """变换注意力 (Transform Attention)，用于连接编码器和解码器"""

    def __init__(self, K, d, bn=False, **kwargs):
        super().__init__(**kwargs)
        self.K = K
        self.d = d
        self.D = K * d
        self.bn = bn

        self.query = FC(self.D, "relu", bn=bn)
        self.key = FC(self.D, "relu", bn=bn)
        self.value = FC(self.D, "relu", bn=bn)
        self.fc = FC([self.D, self.D], ["relu", None], bn=bn)

    def call(self, X, STE_P, STE_Q, training=None):
        # X from encoder: (B, P, N, D)
        # STE_P from encoder: (B, P, N, D)
        # STE_Q from decoder: (B, Q, N, D)

        query = self.query(STE_Q, training=training)  # Query from decoder context
        key = self.key(STE_P, training=training)  # Key from encoder context
        value = self.value(X, training=training)  # Value from encoder output

        B = ops.shape(X)[0]
        P = ops.shape(X)[1]
        Q = ops.shape(STE_Q)[1]
        N = ops.shape(X)[2]

        # --- 优化点 1: 使用 reshape 和 transpose 实现多头注意力 ---
        # Query: (B, Q, N, D) -> (B, K, N, Q, d)
        query = ops.transpose(
            ops.reshape(query, (B, Q, N, self.K, self.d)), (0, 3, 2, 1, 4)
        )
        # Key: (B, P, N, D) -> (B, K, N, P, d)
        key = ops.transpose(
            ops.reshape(key, (B, P, N, self.K, self.d)), (0, 3, 2, 1, 4)
        )
        # Value: (B, P, N, D) -> (B, K, N, P, d)
        value = ops.transpose(
            ops.reshape(value, (B, P, N, self.K, self.d)), (0, 3, 2, 1, 4)
        )

        # Scaled Dot-Product Attention
        # (B, K, N, Q, d) @ (B, K, N, d, P) -> (B, K, N, Q, P)
        # attention_scores = ops.matmul(query, key, transpose_b=True)
        attention_scores = ops.matmul(query, ops.transpose(key, axes=[0, 1, 2, 4, 3]))
        attention_scores = ops.divide(
            attention_scores, ops.sqrt(ops.cast(self.d, attention_scores.dtype))
        )

        attention_weights = ops.softmax(attention_scores, axis=-1)

        # Apply attention
        # (B, K, N, Q, P) @ (B, K, N, P, d) -> (B, K, N, Q, d)
        attended_value = ops.matmul(attention_weights, value)

        # Concatenate heads
        # (B, K, N, Q, d) -> (B, Q, N, K, d) -> (B, Q, N, D)
        concatenated_value = ops.reshape(
            ops.transpose(attended_value, (0, 3, 2, 1, 4)), (B, Q, N, self.D)
        )

        return self.fc(concatenated_value, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"K": self.K, "d": self.d, "bn": self.bn})
        return config


@keras.saving.register_keras_serializable()
class GMAN(keras.Model):
    """
    图多注意力网络 (GMAN) 的最终优化实现。
    利用现代 Keras 3 API 自动处理混合精度和梯度裁剪。
    """

    def __init__(self, args, SE, mean, std, bn=False, **kwargs):
        super().__init__(**kwargs)
        self.args = args
        self.L = args.L
        self.P = args.P
        self.Q = args.Q
        self.T = 24 * 60 // args.time_slot
        D = args.K * args.d

        self.SE = tf.constant(SE, dtype="float32")
        self.mean = tf.constant(float(mean), dtype="float32")
        self.std = tf.constant(float(std), dtype="float32")

        self.fc_x = FC([D, D], ["relu", None], bn=bn)
        self.st_embedding = STEmbedding(D, bn=bn)
        self.encoder = [STAttBlock(args.K, args.d, bn=bn) for _ in range(self.L)]
        self.transform_attention = TransformAttention(args.K, args.d, bn=bn)
        self.decoder = [STAttBlock(args.K, args.d, bn=bn) for _ in range(self.L)]
        self.fc_out = FC([D, 1], ["relu", None], drop=0.1, bn=bn)
        self._build_shape = None

    def build(self, input_shape):
        """构建模型层的形状信息"""
        if not self.built:
            # 构建输入层
            # input_shape: ((batch, P, N), (batch, P+Q, 2))
            # 运行一次前向传播以初始化所有层
            self._build_shape = input_shape
            super().build(input_shape)

    def call(self, inputs, training=None):
        """模型前向传播（保持不变）"""
        X_hist, TE_all = inputs
        X = ops.expand_dims(X_hist, axis=-1)
        X = self.fc_x(X, training=training)
        STE = self.st_embedding(self.SE, TE_all, T=self.T, training=training)
        STE_P = STE[:, : self.P]
        STE_Q = STE[:, self.P :]
        for block in self.encoder:
            X = block(X, STE_P, training=training)
        X_transformed = self.transform_attention(X, STE_P, STE_Q, training=training)
        Y = X_transformed
        for block in self.decoder:
            Y = block(Y, STE_Q, training=training)
        Y_pred = self.fc_out(Y, training=training)
        return ops.squeeze(Y_pred, axis=3)

    # --- ✨ 优化后的 train_step ---
    def train_step(self, data):
        """
        简化的自定义训练步骤。
        混合精度和梯度裁剪由 Keras 自动处理。
        """
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            # 反向标准化预测值以计算真实损失
            # 这是自定义逻辑的核心，必须保留
            y_pred_denorm = ops.add(
                ops.multiply(ops.cast(y_pred, "float32"), self.std), self.mean
            )
            y_true = ops.cast(y, "float32")

            # 直接使用原始 loss。Keras 的优化器包装器会自动处理缩放
            loss = self.compute_loss(y=y_true, y_pred=y_pred_denorm)

        # 直接在原始 loss 上计算梯度
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # 检查梯度是否有效（防止NaN传播）
        gradients = [
            tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) if g is not None else None
            for g in gradients
        ]

        # 应用梯度。优化器会自动处理梯度的反缩放和裁剪
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))  # type: ignore

        # 更新指标（兼容 Keras 3 推荐方式）
        for metric in self.metrics:
            metric.update_state(y_true, y_pred_denorm)

        return {m.name: m.result() for m in self.metrics}

    # --- ✨ 优化后的 test_step ---
    def test_step(self, data):
        """简化的自定义评估步骤"""
        x, y = data

        y_pred = self(x, training=False)

        y_pred_denorm = ops.cast(y_pred, "float32") * self.std + self.mean
        y_true = ops.cast(y, "float32")

        # self.compute_loss 会同时更新 compiled_loss 状态
        self.compute_loss(y=y_true, y_pred=y_pred_denorm)
        for metric in self.metrics:
            metric.update_state(y_true, y_pred_denorm)

        return {m.name: m.result() for m in self.metrics}

    # --- predict_step 已经是最高效的形式，无需改动 ---
    def predict_step(self, data):
        """自定义预测步骤，确保输出是反向标准化的结果"""
        x = data[0] if isinstance(data, tuple) else data
        y_pred = self(x, training=False)
        return ops.add(ops.multiply(ops.cast(y_pred, "float32"), self.std), self.mean)
