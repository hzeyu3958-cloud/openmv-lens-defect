# OpenMV N6 缺陷识别项目

这个项目现在支持两种检测对象，并继续保留原有眼镜片流程：

1. 眼镜片
2. 载玻片

载玻片类别固定为三类：

```text
normal  合格
scratch 划痕
stain   污渍
```

## 1. 当前结构

```text
.
├─ dataset/                         # 眼镜片数据集
├─ dataset_slide/                   # 载玻片数据集
├─ models/
├─ openmv/
│  ├─ n6_usb_image_capture.py
│  ├─ n6_classifier_main.py
│  ├─ n6_usb_slide_capture.py
│  └─ n6_slide_classifier_main.py
├─ training/
│  ├─ train_lens_classifier.py
│  └─ train_slide_classifier.bat
├─ windows_host/
│  └─ lens_defect_host.py
└─ release/
   └─ LensDefectHost.exe
```

## 2. Windows 上位机

上位机仍然只有一个程序：

```text
release/LensDefectHost.exe
```

新增了“检测对象”选择：

- 眼镜片检测
- 载玻片检测

切换后会自动切到对应的默认路径：

- 眼镜片数据集：`dataset/`
- 载玻片数据集：`dataset_slide/`
- 眼镜片模型：`models/lens_defect_classifier_int8.tflite`
- 载玻片模型：`models/slide_defect_classifier_int8.tflite`
- 眼镜片标签：`models/lens_defect_labels.txt`
- 载玻片标签：`models/slide_defect_labels.txt`

说明：

- 眼镜片模式继续保留原来的电脑端 YOLO/二级复核逻辑。
- 载玻片模式默认直接显示 OpenMV 输出的 JSON 结果，不套用镜片专用复核逻辑，避免误修正。

## 3. 数据采集

### 3.1 眼镜片

在 OpenMV IDE 运行：

```text
openmv/n6_usb_image_capture.py
```

上位机切到“眼镜片检测”后，在“采集训练数据”页采集，默认保存到：

```text
dataset/train|val|test/normal|scratch|stain/
```

### 3.2 载玻片

在 OpenMV IDE 运行：

```text
openmv/n6_usb_slide_capture.py
```

这个脚本使用适合载玻片的矩形 ROI。

上位机切到“载玻片检测”后，在“采集训练数据”页采集，默认保存到：

```text
dataset_slide/
├─ train/
│  ├─ normal/
│  ├─ scratch/
│  └─ stain/
├─ val/
└─ test/
```

建议先采：

- `normal` 100 张以上
- `scratch` 100 张以上
- `stain` 100 张以上

## 4. 训练载玻片模型

### 4.1 在上位机里训练

1. 打开 `LensDefectHost.exe`
2. 选择“载玻片检测”
3. 进入“3 训练和部署”
4. 点击“启动本地训练脚本”

它会调用：

```text
training/train_lens_classifier.py
```

并输出：

```text
models/slide_defect_classifier.keras
models/slide_defect_classifier_float.tflite
models/slide_defect_classifier_int8.tflite
models/slide_defect_labels.txt
models/slide_training_summary.json
```

### 4.2 直接用 bat 启动

```bat
training\train_slide_classifier.bat
```

### 4.3 直接用命令行启动

```bat
.venv_training\Scripts\python.exe training\train_lens_classifier.py ^
  --dataset dataset_slide ^
  --output models ^
  --artifact-prefix slide_defect ^
  --summary-name slide_training_summary.json ^
  --epochs 20 ^
  --batch-size 16 ^
  --image-size 128
```

说明：

- 训练入口仍然复用现有分类训练流程。
- 眼镜片默认文件名不变，所以旧流程不会被破坏。
- 当前载玻片模式主走分类模型；原镜片 YOLO 种子导出与训练仍保留给眼镜片模式。

## 5. 把载玻片模型复制到 OpenMV

上位机切到“载玻片检测”后，在“3 训练和部署”页选择：

- 模型：`models/slide_defect_classifier_int8.tflite`
- 标签：`models/slide_defect_labels.txt`
- OpenMV 盘符/目录

点击“复制模型到 OpenMV N6”后，会复制成：

```text
/slide_defect_classifier_int8.tflite
/slide_defect_labels.txt
```

## 6. OpenMV 上运行哪个脚本

### 6.1 载玻片采集

```text
openmv/n6_usb_slide_capture.py
```

### 6.2 载玻片检测

```text
openmv/n6_slide_classifier_main.py
```

这个脚本：

- 使用载玻片专用矩形 ROI
- 每次输出一行 JSON
- 继续支持 `IMG_BEGIN / IMG_END` 画面协议
- JSON 结构保持和原上位机兼容

### 6.3 眼镜片检测

原脚本不变，继续使用：

```text
openmv/n6_classifier_main.py
openmv/n6_lens_defect_rule_main.py
openmv/n6_usb_image_capture.py
```

## 7. 上位机显示结果

载玻片模式下显示结果仍然是：

- 划痕
- 污渍
- 合格

眼镜片模式继续显示原来的眼镜片结果。

## 8. EXE

最终仍然只保留一个 Windows 程序：

```text
release/LensDefectHost.exe
```

如需重新打包：

```bat
windows_host\build_windows_exe.bat
```
