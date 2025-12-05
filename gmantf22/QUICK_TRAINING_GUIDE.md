# GMAN 快速训练指南

## 3 阶段渐进式训练方案

### 问题分析
- 原始配置：L=5, K=8, d=8 → 320维特征，计算量巨大
- batch_size=12 太小，GPU 利用率低
- decay_epoch=5 导致学习率衰减过快

### 阶段 1️⃣：快速验证 (5-15 分钟)
**目标**：快速得到一个能收敛的模型，验证管道是否工作

```bash
# 运行快速训练
python gmantf22/train_quick.py

# 在另一个终端查看 TensorBoard
tensorboard --logdir=logs/fit_quick
```

**快速模式配置**：
- L=2 (原 5)
- K=4 (原 8) 
- d=4 (原 8)
- batch_size=32 (原 12)
- max_epoch=50
- learning_rate=0.002 (更激进)
- enable_xla=False (快速编译)

**预期结果**：
- 5-10 epoch 看到明显 loss 下降
- 20 epoch 达到可用的精度
- 快速验证模型没有 bug

---

### 阶段 2️⃣：中等训练 (30-60 分钟)
**目标**：使用更完整的模型，但不是完整配置

创建 `config_medium.py`：
```python
class GMANConfigMedium(BaseModel):
    L: int = Field(3, description="number of STAtt Blocks")
    K: int = Field(6, description="number of attention heads")
    d: int = Field(6, description="dims of each head attention outputs")
    batch_size: int = Field(24, description="batch size")
    max_epoch: int = Field(80, description="epoch to run")
    learning_rate: float = Field(0.0015, description="initial learning rate")
    decay_epoch: int = Field(8, description="decay epoch")
    enable_xla: bool = Field(True, description="enable XLA")
```

```bash
# 运行中等规模训练
python gmantf22/train_medium.py

# 监控
tensorboard --logdir=logs/fit_medium
```

---

### 阶段 3️⃣：完整训练 (2-4 小时)
**目标**：使用完整配置，从阶段2的权重继续训练（warm start）

```bash
# 方案A: 从头开始完整训练
python gmantf22/train.py

# 方案B: 从阶段2的权重继续（推荐）
# 在 train.py 中添加：
# model.load_weights('./models/GMAN_medium.weights.h5')

tensorboard --logdir=logs/fit
```

---

## 快速对比三种配置

| 配置 | 模型大小 | Batch | Epoch | 耗时 | 用途 |
|------|--------|-------|-------|------|------|
| Quick | 64维 | 32 | 50 | 5-15min | 快速验证 |
| Medium | 108维 | 24 | 80 | 30-60min | 精度调整 |
| Full | 320维 | 12 | 100 | 2-4h | 最优结果 |

---

## 💡 快速训练技巧

### 1. 检查快速模式是否收敛
```
如果快速模式 loss 在 5-10 epoch 快速下降 → ✅ 管道正确
如果快速模式 loss 平缓 → 需要增加 learning_rate
如果快速模式 loss 上升 → 需要减少 learning_rate
```

### 2. 从快速模型导入权重到完整模型
```python
import tensorflow as tf

# 快速模型权重
quick_weights = './models/GMAN_quick.weights.h5'

# 完整模型权重
full_model.load_weights(quick_weights, by_name=True, skip_mismatch=True)
# 这样会加载所有名字匹配的权重（Embedding、前几层等）
# 跳过尺寸不匹配的权重（新增的层）
```

### 3. 使用 TensorBoard 对比三个阶段
```bash
# 同时查看所有三个阶段
tensorboard --logdir=logs/
# 会显示 fit_quick, fit_medium, fit 三个分支
```

### 4. 监控指标
- **loss 下降快** → 学习率合适 ✅
- **loss 平缓** → 增加 learning_rate 或减少 L/K/d
- **loss 波动** → 减少 learning_rate 或增加 batch_size
- **val_loss 上升** → 过拟合，启用早停或增加 dropout

---

## 实际工作流建议

```
1. 运行快速模式 train_quick.py (5 min)
   ├─ 检查 loss 是否下降
   └─ 查看 TensorBoard: tensorboard --logdir=logs/fit_quick

2a. 如果快速模式失败
   └─ 调整 config_quick.py 中的 learning_rate, batch_size, L
   
2b. 如果快速模式成功
   └─ 运行中等模式 train_medium.py (30 min)
   
3. 如果满足中等模式结果
   └─ 继续完整模式或微调参数
   
4. 最终用最佳参数做完整训练
   └─ python train.py
```

---

## 各阶段 Loss 预期

**快速模式**（L=2, K=4, d=4）:
```
Epoch 1:  loss ≈ 50-60
Epoch 5:  loss ≈ 20-30
Epoch 20: loss ≈ 5-10 (可用)
```

**中等模式**（L=3, K=6, d=6）:
```
Epoch 1:  loss ≈ 55-65
Epoch 10: loss ≈ 15-25
Epoch 50: loss ≈ 3-8 (较好)
```

**完整模式**（L=5, K=8, d=8）:
```
Epoch 1:   loss ≈ 60-70
Epoch 20:  loss ≈ 10-20
Epoch 100: loss ≈ 1-3 (最优)
```

---

## 常见问题

**Q: 快速模式 loss 不下降？**
- A: 尝试增加 learning_rate 到 0.003 或 0.004

**Q: 快速模式过拟合明显？**
- A: 减少 L, K, d，或增加 batch_size 到 64

**Q: 从快速模型转换到完整模型？**
- A: 使用 `load_weights(..., by_name=True, skip_mismatch=True)`

**Q: 如何只训练特定 epoch？**
- A: 在 config 中设置 `max_epoch=20`，配合 early_stopping 使用

---

## 总结

✅ **快速验证** (5-15 min) → **中等调整** (30-60 min) → **完整训练** (2-4 h)

每阶段都能看到实际效果，避免在大模型上浪费时间！
