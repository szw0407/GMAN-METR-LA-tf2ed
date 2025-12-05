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
    batch_size: int = Field(12, description="batch size")
    max_epoch: int = Field(100, description="epoch to run")
    patience: int = Field(10, description="patience for early stop")
    learning_rate: float = Field(0.001, description="initial learning rate")
    decay_epoch: int = Field(5, description="decay epoch")
    traffic_file: str = Field("./data/METR-LA/metr-la.h5", description="traffic file")
    SE_file: str = Field(
        "./data/METR-LA/SE(METR).txt", description="spatial embedding file"
    )
    model_file: str = Field(
        "./models/GMAN.weights.h5", description="save the model to disk"
    )
    log_file: str = Field("./log/train.log", description="log file")
    use_mixed_precision: bool = Field(True, description="use mixed precision training")
    enable_xla: bool = Field(True, description="enable XLA JIT for performance")
