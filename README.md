# GMAN: A Graph Multi-Attention Network for Traffic Prediction (AAAI-2020)

<p align="center">
  <img width="600" height="450" src=./figure/GMAN.png>
</p>

This is the implementation of Graph Multi-Attention Network in the following paper: \
Chuanpan Zheng, Xiaoliang Fan*, Cheng Wang, and Jianzhong Qi. "[GMAN: A Graph Multi-Attention Network for Traffic Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/5477)", Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-20), 2020, 34(01): 1234-1241.

## Data

The datasets are available at [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g), provided by [DCRNN](https://github.com/liyaguang/DCRNN), and should be put into the corresponding `data/` folder.

## Requirements

See Migration Notes for details.

## 配置参数

在 `gmantf22/config.py` 中可以配置模型参数。

## 启动

在 `gmantf22/train.py` 中启动训练。请注意根目录必须是 `GMAN/`而不是在 `gmantf22/` 中。

## 可视化与 TensorBoard

训练过程中会自动记录 TensorBoard 日志和训练曲线图片。

### 启动 TensorBoard

运行：

```fish
tensorboard --logdir logs/fit
```

然后在浏览器中访问 <http://localhost:6006> 查看训练过程。

训练结束后，loss/metric 曲线图片会自动保存在对应日志目录下（如 `logs/fit/20251015-XXXXXX/training_curves.png`）。

---

## Results

<p align="center">
  <img width="900" height="400" src=./figure/results.png>
</p>
