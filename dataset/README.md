# 训练数据集目录

Windows 上位机会把 OpenMV N6 通过 USB 发来的图片保存到这里。

推荐目录结构：

```text
dataset/
├── train/
│   ├── normal/
│   ├── scratch/
│   ├── dust/
│   ├── stain/
│   ├── coating_damage/
│   ├── crack/
│   ├── edge_damage/
│   └── unknown/
├── val/
└── test/
```

上位机里点击“初始化分类文件夹”会自动创建这些目录。
