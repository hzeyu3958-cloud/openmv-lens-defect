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

采集图片时，上位机会自动维护：

- `metadata.csv`：每张图片的采集时间、类别、train/val/test 归属、尺寸、文件大小、亮度、对比度和质量提示。
- 最近采集预览：用于现场确认镜片位置、曝光和清晰度。
- 数据集体检：提示每类样本是否太少、是否缺少验证/测试集、类别是否明显不均衡。

建议先让每个类别至少采集 100 张，效果稳定后再扩到每类 300 张以上。
