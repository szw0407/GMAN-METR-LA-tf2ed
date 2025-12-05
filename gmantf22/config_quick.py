"""
快速训练配置 - 用于快速验证模型可行性
在这个配置下训练 10-20 epoch，loss 会快速下降
然后可以过渡到完整配置进行微调
"""
from pydantic import BaseModel, Field

class GMANConfigQuick(BaseModel):
    time_slot: int = Field(5, description="time interval")
    P: int = Field(12, description="history steps")
    Q: int = Field(12, description="prediction steps")
    # --- 简化模型架构 ---
    L: int = Field(2, description="number of STAtt Blocks - REDUCED from 5")
    K: int = Field(4, description="number of attention heads - REDUCED from 8")
    d: int = Field(4, description="dims of each head attention outputs - REDUCED from 8")
    
    train_ratio: float = Field(0.7, description="training set ratio")
    val_ratio: float = Field(0.1, description="validation set ratio")
    test_ratio: float = Field(0.2, description="testing set ratio")
    # --- 加大 batch size ---
    batch_size: int = Field(32, description="batch size - INCREASED from 12")
    # --- 短期训练 ---
    max_epoch: int = Field(50, description="epoch to run - REDUCED for quick testing")
    patience: int = Field(8, description="patience for early stop")
    # --- 更激进的学习率和衰减 ---
    learning_rate: float = Field(0.002, description="initial learning rate - INCREASED")
    decay_epoch: int = Field(10, description="decay epoch - INCREASED to let LR stay high longer")
    
    traffic_file: str = Field("./data/METR-LA/metr-la.h5", description="traffic file")
    SE_file: str = Field(
        "./data/METR-LA/SE(METR).txt", description="spatial embedding file"
    )
    model_file: str = Field(
        "./models/GMAN_quick.weights.h5", description="save the model to disk"
    )
    log_file: str = Field("./log/train_quick.log", description="log file")
    use_mixed_precision: bool = Field(True, description="use mixed precision training")
    enable_xla: bool = Field(False, description="disable XLA to speed up training during quick mode")
