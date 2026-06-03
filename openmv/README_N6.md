# OpenMV N6 脚本说明

项目现在分成两条链路：

1. 眼镜片
2. 载玻片

## 1. 眼镜片

采集：

```text
openmv/n6_usb_image_capture.py
```

规则检测：

```text
openmv/n6_lens_defect_rule_main.py
```

模型检测：

```text
openmv/n6_classifier_main.py
```

模型文件：

```text
/lens_defect_classifier_int8.tflite
/lens_defect_labels.txt
```

## 2. 载玻片

采集：

```text
openmv/n6_usb_slide_capture.py
```

模型检测：

```text
openmv/n6_slide_classifier_main.py
```

模型文件：

```text
/slide_defect_classifier_int8.tflite
/slide_defect_labels.txt
```

说明：

- 载玻片脚本使用专用矩形 ROI。
- 输出协议继续兼容 Windows 上位机。
- 每次输出一行 JSON。
- 开启画面时继续使用 `IMG_BEGIN / IMG_END`。

## 3. 串口

默认主要走 USB VCP，适合接 Windows 上位机。

如果要接蓝牙串口模块或单片机，可在脚本里打开：

```python
ENABLE_UART_OUTPUT = True
UART_ID = 3
UART_BAUDRATE = 115200
```

N6 UART3 引脚：

```text
P4 = TX
P5 = RX
GND 共地
```
