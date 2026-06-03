# 载玻片数据集

默认训练目录结构：

```text
dataset_slide/
├─ train/
│  ├─ normal/
│  ├─ scratch/
│  └─ stain/
├─ val/
│  ├─ normal/
│  ├─ scratch/
│  └─ stain/
└─ test/
   ├─ normal/
   ├─ scratch/
   └─ stain/
```

说明：

- `normal`：合格载玻片
- `scratch`：划痕
- `stain`：污渍

Windows 上位机切到“载玻片检测”模式后，采集页默认就会写入这个目录。
