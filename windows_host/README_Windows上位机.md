# Windows 上位机说明

当前上位机只有一个程序：

```text
release/LensDefectHost.exe
```

它现在支持两种检测对象：

- 眼镜片检测
- 载玻片检测

## 1. 先切检测对象

主界面顶部新增“检测对象”下拉框。

切换后会自动切换默认资源：

- 数据集目录
- 模型文件
- 标签文件
- 训练结果摘要文件

默认对应关系：

```text
眼镜片:
  dataset/
  models/lens_defect_classifier_int8.tflite
  models/lens_defect_labels.txt

载玻片:
  dataset_slide/
  models/slide_defect_classifier_int8.tflite
  models/slide_defect_labels.txt
```

## 2. 采集训练数据

眼镜片采集脚本：

```text
openmv/n6_usb_image_capture.py
```

载玻片采集脚本：

```text
openmv/n6_usb_slide_capture.py
```

在上位机“2 采集训练数据”页里：

1. 先切检测对象
2. 选择类别
3. 点击“拍一张保存到电脑”或“开始自动采集”

## 3. 训练模型

在“3 训练和部署”页点击“启动本地训练脚本”。

说明：

- 眼镜片模式输出原文件名，兼容旧流程。
- 载玻片模式会输出 `slide_defect_*` 文件。
- 当前载玻片模式主走分类训练流程。

## 4. 复制模型到 OpenMV

眼镜片模式会复制成：

```text
/lens_defect_classifier_int8.tflite
/lens_defect_labels.txt
```

载玻片模式会复制成：

```text
/slide_defect_classifier_int8.tflite
/slide_defect_labels.txt
```

## 5. OpenMV 端脚本

眼镜片模型检测：

```text
openmv/n6_classifier_main.py
```

载玻片模型检测：

```text
openmv/n6_slide_classifier_main.py
```

## 6. 打包 exe

```bat
windows_host\build_windows_exe.bat
```

打包后仍然只有一个 exe：

```text
release/LensDefectHost.exe
```
