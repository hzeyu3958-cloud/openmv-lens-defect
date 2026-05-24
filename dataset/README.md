# 训练数据集目录

当前项目只训练三类：

```text
dataset/
├── train/
│   ├── normal/
│   ├── scratch/
│   └── stain/
├── val/
│   ├── normal/
│   ├── scratch/
│   └── stain/
└── test/
    ├── normal/
    ├── scratch/
    └── stain/
```

类别含义：

- `normal`：正常镜片
- `scratch`：划痕
- `stain`：污渍

建议先保证每类至少 `train 40 / val 10 / test 10`，效果稳定后再扩到每类 100 张以上。
