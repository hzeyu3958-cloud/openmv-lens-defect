# OpenMV N6 使用说明

这份说明只针对星瞳/OpenMV N6 + PAG7936 版本。

## 到货前可以先准备

1. 运行 Windows 上位机，确认“模拟一条”能正常显示 JSON。
2. 准备数据集目录结构，或直接用上位机点击“初始化分类文件夹”。
3. 准备固定夹具、黑色背景和稳定补光。眼镜片缺陷检测对光源比模型更敏感。
4. 先用 `n6_lens_tracker_main.py` 调镜片跟踪，再用 `n6_lens_defect_rule_main.py` 熟悉 ROI 和阈值。

实时跟踪镜片不需要再加单片机。OpenMV N6 自己可以完成摄像头采集、镜片跟踪、缺陷检测和 JSON 输出。只有你要控制电机、舵机、传送带、气缸分拣等外设时，才需要另加 STM32/Arduino/ESP32 或 PLC。

## 到货后第一步

在 OpenMV IDE 里先运行：

```text
openmv/n6_lens_tracker_main.py
```

这个脚本只做实时跟踪。它会在画面里给镜片画框和中心十字，并通过 USB VCP 输出：

```json
{
  "model": "lens_tracker",
  "lens": {
    "found": true,
    "x": 40,
    "y": 30,
    "w": 240,
    "h": 180,
    "cx": 160,
    "cy": 120,
    "confidence": 0.82,
    "source": "tracked"
  }
}
```

跟踪稳定后，再运行：

```text
openmv/n6_lens_defect_rule_main.py
```

这个脚本不依赖模型。它会：

- 使用 `csi.CSI()` 初始化 PAG7936。
- 以 QVGA 灰度图实时跟踪镜片。
- 用跟踪到的镜片 ROI 运行传统缺陷检测。
- 在画面上画出镜片框、中心点和缺陷框。
- 通过 USB VCP 输出一行一条 JSON，Windows 上位机可以直接接收。

先调这些参数：

```python
LENS_ROI = (40, 30, 240, 180)
ENABLE_LENS_TRACKING = True
TRACK_MIN_CONTRAST_DELTA = 12
TRACK_MIN_AREA = 900
MIN_CONTRAST_DELTA = 18
STD_FACTOR = 1.35
SCRATCH_MIN_ASPECT_RATIO = 5.0
SCRATCH_MIN_LENGTH = 24
```

如果镜片框跟丢：

- 黑色背景 + 侧光或环形补光会更稳定。
- 误抓到夹具时，缩小 `TRACK_SEARCH_ROI`，让搜索范围只覆盖镜片可能出现的区域。
- 框太小或太碎时，降低 `TRACK_MIN_AREA` 和 `TRACK_MIN_PIXELS`。
- 框把背景也吃进去时，增大 `TRACK_MIN_CONTRAST_DELTA` 或 `TRACK_MIN_AREA`。

## 采集训练图片

在 OpenMV IDE 里运行：

```text
openmv/n6_usb_image_capture.py
```

然后在 Windows 上位机“2 采集训练数据”页点击“拍一张保存到电脑”。

建议每个类别先采：

```text
normal: 100 张以上
scratch: 100 张以上
dust: 100 张以上
stain: 100 张以上
```

当前上位机和 OpenMV 脚本只保留 `normal/scratch/dust/stain` 四类，类别少时训练更容易稳定。

## 训练后运行模型

电脑训练完成后，把下面两个文件复制到 OpenMV N6 的 USB 盘根目录：

```text
lens_defect_classifier_int8.tflite
lens_defect_labels.txt
```

训练脚本默认使用 `tiny_depthwise` 轻量模型，流程是电脑训练、导出 INT8 TFLite、再复制给 N6。N6 端只负责推理，不在板子上训练。

然后在 OpenMV IDE 里运行：

```text
openmv/n6_classifier_main.py
```

模型脚本会自动在 `/`、`/flash`、`/sdcard` 搜索模型和标签文件。

## USB 与蓝牙输出

默认脚本打开 USB VCP，适合接 Windows 上位机。`n6_lens_defect_rule_main.py` 和 `n6_classifier_main.py` 会发送 JSON，也会按 `IMG_BEGIN/IMG_END` 协议发送 JPEG 画面供上位机显示。

如果要接 HC-05/HC-06 蓝牙串口模块，或把识别结果预留给单片机串口接收，在脚本里打开：

```python
ENABLE_UART_OUTPUT = True
UART_ID = 3
UART_BAUDRATE = 115200
```

N6 UART3 引脚：

```text
N6 P4 TX -> 蓝牙模块 RX
N6 P5 RX -> 蓝牙模块 TX
N6 GND   -> 蓝牙模块 GND
```

注意 N6 GPIO 不是 5V 容忍，串口信号请按 3.3V 电平处理。

## 参考资料

- OpenMV N6 官方文档: https://docs.openmv.io/openmvcam/quickref/openmv-n6.html
- OpenMV N6 产品规格: https://openmv.io/products/openmv-n6
- OpenMV 官方 GitHub: https://github.com/openmv/openmv
- OpenMV ml 模块文档: https://docs.openmv.io/library/omv.ml.html
