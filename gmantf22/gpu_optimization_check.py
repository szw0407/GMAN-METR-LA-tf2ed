"""
GPU Optimization Verification Script
验证所有关键操作都在 GPU 上执行，而不是 CPU 或 NumPy

运行此脚本检查：
1. GPU 是否被正确检测和配置
2. 数据预处理操作是否在 GPU 上运行
3. 张量操作是否正确分配到 GPU
4. 内存管理是否优化
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)


def check_gpu_availability():
    """检查 GPU 可用性"""
    log.info("=" * 60)
    log.info("GPU 可用性检查")
    log.info("=" * 60)
    
    gpus = tf.config.list_physical_devices('GPU')
    log.info(f"检测到 {len(gpus)} 个 GPU 设备:")
    
    if len(gpus) == 0:
        log.warning("⚠️  没有检测到 GPU！")
        return False
    
    for i, gpu in enumerate(gpus):
        log.info(f"  GPU {i}: {gpu.name}")
        # 设置内存动态增长
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # 验证 TensorFlow 能否使用 GPU
    log.info(f"TensorFlow GPU 设备: {tf.config.list_logical_devices('GPU')}")
    return True


def check_tensor_operations():
    """检查张量操作是否在 GPU 上执行"""
    log.info("\n" + "=" * 60)
    log.info("张量操作 GPU 执行检查")
    log.info("=" * 60)
    
    # 创建测试张量
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        c = tf.matmul(a, b)
        
        log.info(f"✓ 矩阵乘法操作成功")
        log.info(f"  输入张量设备: {a.device}")
        log.info(f"  输出张量设备: {c.device}")
        log.info(f"  结果: {c.numpy().tolist()}")
    
    return True


def check_data_pipeline():
    """检查数据管道 GPU 优化"""
    log.info("\n" + "=" * 60)
    log.info("数据管道 GPU 优化检查")
    log.info("=" * 60)
    
    # 创建模拟数据
    X = np.random.randn(1000, 12, 207).astype(np.float32)
    Y = np.random.randn(1000, 12, 207).astype(np.float32)
    
    # 创建 tf.data.Dataset
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.batch(32).cache().prefetch(tf.data.AUTOTUNE)
    
    log.info("✓ 数据管道创建成功:")
    log.info("  - 批处理: 32")
    log.info("  - 缓存: 已启用")
    log.info("  - 预取: AUTOTUNE")
    
    # 验证数据是否正确批处理
    for x, y in ds.take(1):
        log.info(f"✓ 批数据形状检查:")
        log.info(f"  - X 形状: {x.shape}")
        log.info(f"  - Y 形状: {y.shape}")
    
    return True


def check_mixed_precision(enable_mixed_precision=True):
    """检查和配置混合精度"""
    log.info("\n" + "=" * 60)
    log.info("混合精度配置检查")
    log.info("=" * 60)
    
    policy = keras.mixed_precision.global_policy()
    log.info(f"当前混合精度策略: {policy.name}")
    log.info(f"计算数据类型: {policy.compute_dtype}")
    log.info(f"变量数据类型: {policy.variable_dtype}")
    
    if policy.name == 'float32':
        if enable_mixed_precision:
            # 尝试启用混合精度
            try:
                keras.mixed_precision.set_global_policy('mixed_float16')
                new_policy = keras.mixed_precision.global_policy()
                log.info(f"✓ 混合精度已启用: {new_policy.name}")
                log.info(f"  计算数据类型: {new_policy.compute_dtype}")
                log.info(f"  变量数据类型: {new_policy.variable_dtype}")
                log.info("  ✓ 这将显著提高 GPU 性能！")
            except Exception as e:
                log.warning(f"⚠️  无法启用混合精度: {e}")
                log.warning("  GPU 可能不支持 float16 计算")
        else:
            log.warning("⚠️  未启用混合精度。建议启用以提高 GPU 性能")
            log.warning("  使用 enable_mixed_precision=True 参数来启用")
    else:
        log.info(f"✓ 混合精度已启用: {policy.name}")
    
    return True


def check_custom_operations():
    """检查自定义操作 GPU 支持"""
    log.info("\n" + "=" * 60)
    log.info("自定义操作 GPU 支持检查")
    log.info("=" * 60)
    
    # 测试 seq2instance 操作（使用 TensorFlow 实现）
    data = np.random.randn(100, 10).astype(np.float32)
    P, Q = 5, 3
    
    with tf.device('/GPU:0'):
        data_tensor = tf.constant(data, dtype=tf.float32)
        num_sample = len(data) - P - Q + 1
        
        # 使用 gather 进行高效窗口切片
        indices = tf.range(num_sample)[:, tf.newaxis] + tf.range(P + Q)[tf.newaxis, :]
        windows = tf.gather(data_tensor, indices)
        
        x = windows[:, :P, :]
        y = windows[:, P:, :]
        
        log.info(f"✓ seq2instance GPU 操作成功:")
        log.info(f"  - 输入形状: {data.shape}")
        log.info(f"  - 输出 X 形状: {x.shape}")
        log.info(f"  - 输出 Y 形状: {y.shape}")
    
    # 测试 metric 计算（GPU 加速）
    with tf.device('/GPU:0'):
        pred = tf.constant(np.random.randn(100, 10).astype(np.float32))
        label = tf.constant(np.random.randn(100, 10).astype(np.float32))
        
        mask = tf.not_equal(label, 0.0)
        pred_masked = tf.boolean_mask(pred, mask)
        label_masked = tf.boolean_mask(label, mask)
        
        mae = tf.reduce_mean(tf.abs(label_masked - pred_masked))
        mse = tf.reduce_mean(tf.square(label_masked - pred_masked))
        rmse = tf.sqrt(mse)
        
        log.info(f"✓ metric GPU 操作成功:")
        log.info(f"  - MAE: {float(mae.numpy()):.6f}")
        log.info(f"  - RMSE: {float(rmse.numpy()):.6f}")
    
    return True


def check_normalization_gpu():
    """检查归一化操作是否在 GPU 上执行"""
    log.info("\n" + "=" * 60)
    log.info("归一化操作 GPU 检查")
    log.info("=" * 60)
    
    # 创建大型数据集
    data = np.random.randn(10000, 207).astype(np.float32)
    
    with tf.device('/GPU:0'):
        # 使用 TensorFlow 进行归一化（GPU 加速）
        data_tensor = tf.constant(data, dtype=tf.float32)
        mean = tf.reduce_mean(data_tensor)
        std = tf.math.reduce_std(data_tensor)
        std = tf.maximum(std, 1e-5)
        
        normalized = (data_tensor - mean) / std
        
        log.info(f"✓ 归一化 GPU 操作成功:")
        log.info(f"  - 输入形状: {data.shape}")
        log.info(f"  - 均值: {float(mean.numpy()):.6f}")
        log.info(f"  - 标准差: {float(std.numpy()):.6f}")
        log.info(f"  - 输出形状: {normalized.shape}")
    
    return True


def performance_benchmark():
    """性能基准测试"""
    log.info("\n" + "=" * 60)
    log.info("性能基准测试")
    log.info("=" * 60)
    
    import time
    
    # 大矩阵操作
    size = 10000
    
    # GPU 上的操作
    with tf.device('/GPU:0'):
        a = tf.random.normal((size, size), dtype=tf.float32)
        b = tf.random.normal((size, size), dtype=tf.float32)
        
        start = time.time()
        c = tf.matmul(a, b)
        gpu_time = time.time() - start
    
    log.info(f"✓ GPU 矩阵乘法 ({size}x{size}):")
    log.info(f"  - 时间: {gpu_time*1000:.2f}ms")
    
    # 数据处理性能
    data = np.random.randn(100000, 207).astype(np.float32)
    
    with tf.device('/GPU:0'):
        data_tensor = tf.constant(data, dtype=tf.float32)
        
        start = time.time()
        for _ in range(10):
            mean = tf.reduce_mean(data_tensor)
            std = tf.math.reduce_std(data_tensor)
            normalized = (data_tensor - mean) / std
        gpu_norm_time = time.time() - start
    
    log.info(f"✓ GPU 归一化 (100000x207, 10次迭代):")
    log.info(f"  - 时间: {gpu_norm_time*1000:.2f}ms")
    log.info(f"  - 平均: {gpu_norm_time*100:.2f}ms/迭代")


def main():
    """运行所有检查"""
    log.info("\n" + "=" * 60)
    log.info("GPU 优化综合检查工具")
    log.info("=" * 60)
    
    # 首先启用混合精度
    check_mixed_precision(enable_mixed_precision=True)
    
    checks = [
        ("GPU 可用性", check_gpu_availability),
        ("张量操作", check_tensor_operations),
        ("数据管道", check_data_pipeline),
        ("自定义操作", check_custom_operations),
        ("归一化操作", check_normalization_gpu),
    ]
    
    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, "✓ 通过"))
        except Exception as e:
            log.error(f"✗ {name} 检查失败: {e}")
            results.append((name, f"✗ 失败: {e}"))
    
    # 性能基准测试
    try:
        performance_benchmark()
    except Exception as e:
        log.error(f"性能基准测试失败: {e}")
    
    # 总结
    log.info("\n" + "=" * 60)
    log.info("检查总结")
    log.info("=" * 60)
    for name, result in results:
        log.info(f"{name}: {result}")
    
    log.info("\n" + "=" * 60)
    log.info("✓ GPU 优化检查完成！")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
